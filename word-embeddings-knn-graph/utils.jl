using LinearAlgebra, JSON3

function parse_vector(_type, line::String)
	map(s -> parse(_type, s), split(line))
end

function read_token(f, buffer)
	empty!(buffer)
	while !eof(f)
		c = read(f, Char)
		if c in (' ', '\t', '\n')
			length(buffer) > 0 && break
		else
			push!(buffer, c)
		end
	end

	String(buffer)
end

function read_vector(f, dim, buffer::Vector{Char})
	w = read_token(f, buffer)
	v = Vector{Float32}(undef, dim)
	
	for i in 1:dim
		v[i] = parse(Float32, read_token(f, buffer))
	end
	
	w, normalize!(v)
end

function read_embedding(filename)
	buffer = Vector{Char}()

	open(filename) do f
		n = parse(Int, read_token(f, buffer))
		dim = parse(Int, read_token(f, buffer))

		@show n, dim, pwd()
		voc = Vector{String}(undef, n)
		vecs = Vector{Vector{Float32}}(undef, n)

		for i in 1:n
			voc[i], vecs[i] = read_vector(f, dim, buffer)
			if (i % 100000) == 1
				@info "reading word embedding $i, $(Dates.now())"
			end
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