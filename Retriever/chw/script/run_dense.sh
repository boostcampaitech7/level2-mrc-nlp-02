
export PYTHONPATH=$PYTHONPATH:.
python Retriever/chw/dense_test.py --dataset_name ./data/train_dataset --data_path ./data --context_path wikipedia_documents.json --model_name_or_path klue/bert-base --method bert --device cuda --topk 10