
export PYTHONPATH=$PYTHONPATH:.
python Retriever/chw/hybrid_test.py --dataset_name ./data/train_dataset --data_path ./data --context_path wikipedia_documents.json --model_name_or_path klue/bert-base --dense_method bert --device cuda --mode eval --sparse_method bm25 --topk 5