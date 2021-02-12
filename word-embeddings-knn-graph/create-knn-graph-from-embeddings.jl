using SimilaritySearch, Dates
include("utils.jl")

function main(embeddingfile, k=11)
	indexfile = replace(embeddingfile, ".vec" => "") * ".index"
	knnfile = indexfile * ".$(k)nn-graph"

	voc, index = if isfile(indexfile)
		loadindex(indexfile)
	else
		X = read_embedding(embeddingfile)
		index = SearchGraph(NormalizedCosineDistance(), X.vecs; parallel=true, firstblock=50000, block=5000)
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
	println(stderr, "usage example: we=embedding-file.vec k=3 julia -t32 --project=. create-knn-graph-from-embeddings.jl")
	embeddingfile = get(ENV, "we", "")
	k = parse(Int, get(ENV, "k", "11"))
	@assert endswith(embeddingfile, ".vec")
	@assert k > 1
	main(embeddingfile, k)
end