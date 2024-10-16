
export PYTHONPATH=$PYTHONPATH:.
python Retriever/chw/sparse_test.py --dataset_name ./data/train_dataset --data_path ./data --context_path wikipedia_documents.json --model_name_or_path bert-base-multilingual-cased --use_faiss false