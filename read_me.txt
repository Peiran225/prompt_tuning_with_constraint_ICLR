Our code makes use the PEFT package on the hugging face https://huggingface.co/docs/peft/index.
To test our idea, we create a new trainer called my trainer. In my_trainer.py, we add a compute_aux_loss function to compute our regularizer.
Some additional aguments are added:
-- layer==-1 is the original soft prompt tuning. 
-- layer==-2 is the L2 norm of the  word embeddings 
-- layer==i  wiht i>=0 uses the L2 norm with the output of the ith transformer layer.
--particular_layer only run our optimization problem difined with the ith layer. 
--similarity currently has three choices: "L2", "L2_LM" "L2_LM_prompt_only". "L2" corresponds to the L2 squared norm of the word embedding.  "L2_prompt_only"/"L2_LM" is the second/last model in Figure 2 of the paper.
--hook_layer defined the layer that gives us the outputs of the transformers.
--baseline_only when use this, please set --num_of_initial_text=1