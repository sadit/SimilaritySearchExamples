using SimilaritySearch, Embeddings, LinearAlgebra, TextSearch, JLD2, Dates

function allknn(index::SearchGraph, k)
	IDX = [copy(index) for i in 1:Threads.nthreads()]
	KNN = [KnnResult(k) for i in eachindex(index)]

	Threads.@threads for i in eachindex(KNN)
		I = IDX[Threads.threadid()]
		search(I, index[i], KNN[i]; hints=keys(I.links[i]))

		if rand() < 0.0001
			@info "i=$i, n=$(length(KNN)), $(Dates.now())"
		end
	end

	KNN
end

function read_embeddings()
    E = load_embeddings(FastText_Text{:es})
    E.vocab, [normalize!(E.embeddings[:, i]) for i in 1:size(E.embeddings, 2)]
end

function create_semindex(db)
	index = SearchGraph(; dist=NormalizedCosineDistance())
	delete!(index.callbacks, :parameters)
	@time append!(index, db; parallel=true, parallel_firstblock=200_000, parallel_block=100_000)
	index
end

function main(k)
	@info "reading embeddings"	
	vocab, db = read_embeddings()
	@info "creating index on embedding $(size(db))"
	index = create_semindex(db)
	@info "save searchgraph"
	jldsave("searchgraph-ft-es.jld2", index=index)
	@info "computing all knn k=$k"
	knn=allknn(index, k)
	@info "saving allknn"
	jldsave("allknn-ft-es.k=$k.jld2", knn=knn)
	(index=index, knn=knn)
end

if !isinteractive()
	main(16)
end
