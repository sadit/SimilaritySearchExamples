using SimilaritySearch, LinearAlgebra
using JSON3

function parse_vector(_type, line)
	map(s -> parse(_type, s), split(line))
end

function read_embedding(filename)
	open(filename) do f
		n, dim = parse_vector(Int, readline(f))
		@show n, dim, pwd()
		voc = Vector{String}(undef, n)
		vecs = Vector{Vector{Float32}}(undef, n)

		for (i, line) in enumerate(eachline(f))
			#i == 1000 && break
			arr = split(line)
			voc[i] = arr[1]
			v = @view arr[2:end]
			vecs[i] = normalize!([parse(Float32, s) for s in v])
		end

		@assert n == length(voc) && n == length(vecs)
		(voc=voc, vecs=vecs)
	end
end

function saveindex(filename, voc, index)
	open(filename * ".voc", "w") do f
		println(f, typeof(voc))
		JSON3.write(f, voc)
	end

	open(filename, "w") do f
		println(f, typeof(index))
		JSON3.write(f, index)
	end
end

function loadindex(filename)
	voc = open(filename * ".voc") do f
		_type = eval(Meta.parse(readline(f)))
		JSON3.read(f, _type)		
	end

	index = open(filename) do f
		_type = eval(Meta.parse(readline(f)))
		JSON3.read(f, _type)
	end

	voc, index
end

function main(embeddingfile, k=11)
	# embedding source: https://github.com/dccuchile/spanish-word-embeddings/blob/master/emb-from-suc.md
	indexfile = replace(embeddingfile, ".vec" => "") * ".index"
	knnfile = indexfile * ".$(k)nn-graph"

	voc, index = if isfile(indexfile)
		loadindex(indexfile)
	else
		X = read_embedding(embeddingfile)
		index = SearchGraph(NormalizedCosineDistance(), X.vecs; parallel=true)
		@assert length(X.voc) == length(index.db)
		saveindex(indexfile, X.voc, index)
		X.voc, index
	end

	I = [copy(index) for i in 1:Threads.nthreads()]
	G = Dict{String,Tuple}()
	sizehint!(G, length(voc))
	@show length(voc) length(index.db) length(G), length(I[1].res), maxlength(I[1].res)
	@time Threads.@threads for i in 1:length(voc)
		tid = Threads.threadid()
		I[tid].search_algo.hints = I[tid].links[i]
		res = search(I[tid], index.db[i], k)
		G[voc[i]] = ([voc[p.id] for p in res], [p.dist for p in res])
		#G[voc[i]] = [p for p in res]
	end

	@info "saving graph"

	open(knnfile, "w") do f
		#JSON3.pretty(f, JSON3.write(G))
		JSON3.write(f, G)
	end
end

if !isinteractive()
	embeddingfile = get(ENV, "we", "")
	k = parse(Int, get(ENV, "k", "11"))
	@assert endswith(embeddingfile, ".vec")
	@assert k > 1
	#embeddingfile = "embeddings-m-model.vec"  #"fasttext-spanish-xs.vec"
	main(embeddingfile, k)
end