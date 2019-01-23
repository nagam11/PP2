
# First clone Google Seq2Seq model: 

    ``` git clone https://github.com/google/seq2seq.git
        cd seq2seq

# Copy files from this directory to seq2seq directory

# Install package and dependencies:

	``` pip3 install -e .

if you intend to use a GPU:

    ``` pip3 install tensorflow-gpu

otherwise:

    ``` pip3 install tensorflow

# Then export envars:

	``` export VOCAB_SOURCE=${PATH_TO_S2S_DATA}/nmt_data/train/vocab.sources.txt
		export VOCAB_TARGET=${PATH_TO_S2S_DATA}/nmt_data/train/vocab.targets.txt
		export TRAIN_SOURCES=${PATH_TO_S2S_DATA}/nmt_data/train/sources.txt
		export TRAIN_TARGETS=${PATH_TO_S2S_DATA}/nmt_data/train/targets.txt
		export DEV_SOURCES=${PATH_TO_S2S_DATA}/nmt_data/dev/sources.txt
		export DEV_TARGETS=${PATH_TO_S2S_DATA}/nmt_data/dev/targets.txt
		export TEST_SOURCES=${PATH_TO_S2S_DATA}/nmt_data/test/sources_cl.txt
		export TEST_TARGETS=${PATH_TO_S2S_DATA}/nmt_data/test/targets_cl.txt
		export DEV_TARGETS_REF=${PATH_TO_S2S_DATA}/nmt_data/dev/targets.txt
		export MODEL_DIR=${PATH_TO_S2S}/nmt_tutorial
		export PRED_DIR=${MODEL_DIR}/pred
		
		export TRAIN_STEPS=50000

# Create a folder structure:

	``` mkdir -p $MODEL_DIR
		mkdir -p $PRED_DIR

# To train, run:

	``` python3 -m bin.train \
			--config_paths="
			  ./example_configs/nmt_medium.yml,
			  ./example_configs/train_seq2seq.yml,
			  ./example_configs/text_metrics_bpe.yml" \
			--model_params "
			  vocab_source: $VOCAB_SOURCE
			  vocab_target: $VOCAB_TARGET" \
			--input_pipeline_train "
			class: ParallelTextInputPipeline
			params:
			  source_files:
			    - $TRAIN_SOURCES
			  target_files:
			    - $TRAIN_TARGETS" \
			--input_pipeline_dev "
			class: ParallelTextInputPipeline
			params:
			   source_files:
			    - $DEV_SOURCES
			   target_files:
			    - $DEV_TARGETS" \
			--batch_size 32 \
			--train_steps $TRAIN_STEPS \
			--output_dir $MODEL_DIR

# You can use TensorBoard to vuisualize training process:

	``` tensorboard --logdir $MODEL_DIR

# To predict, run:
  
	``` python3 -m bin.infer \
			--tasks "
				- class: DecodeText" \
			--model_dir $MODEL_DIR \
			--input_pipeline "
				class: ParallelTextInputPipeline
				params:
	  				source_files:
	    				- $TEST_SOURCES" \
			>  ${PRED_DIR}/predictions.txt

# To get the summary, run:

    ``` python3 format_preds.py
