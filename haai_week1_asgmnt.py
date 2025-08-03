"""
This program is build with Flan-T5-XL LLM to be able to determine output of a MCQ question with four options. 

> It accepts five parameters provided as a command line input. 
> The first input represents the question and the next four input are the options. 
> The output should be the option number: A/B/C/D 
> Output should be upper-case
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Blue" "Green" "Red" "Yellow"
Terminal Output: A

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Blue" "Red" "Yellow"
Terminal Output: B

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Blue" "Yellow"
Terminal Output: C

 Terminal Input: python template.py "What color is the sky on a clear, sunny day?" "Green" "Red" "Yellow" "Blue"
Terminal Output: D

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

def llm_function(model,tokenizer,q,a,b,c,d):
    '''
    The steps are given for your reference:

    1. Properly formulate the prompt as per the question - which should output either 'YES' or 'NO'. The output must always be upper-case. You may post-process to get the desired output.
    2. Tokenize the prompt
    3. Pass the tokenized prompt to the model get output in terms of logits since the output is deterministic.  
    4. Extract the correct option from the model.
    5. Clean output and return.
    6. Output is case-sensative: A,B,C or D
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    prompt = (f"Question: {q}\n" f"Options:\n" f"A. {a}\n"  f"B. {b}\n" f"C. {c}\n"  f"D. {d}\n" f"Answer with the letter of the correct option." )
    A = tokenizer.encode("A", return_tensors="pt", add_special_tokens=False)
    B = tokenizer.encode("B", return_tensors="pt", add_special_tokens=False)
    C = tokenizer.encode("C", return_tensors="pt", add_special_tokens=False)
    D = tokenizer.encode("D", return_tensors="pt", add_special_tokens=False)
   
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(tokenized_prompt, do_sample=False,  top_p=None, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
    logit_stack = torch.stack(outputs.scores, dim=1)
    logit_a = logit_stack[0][0][A].item()
    logit_b = logit_stack[0][0][B].item()
    logit_c = logit_stack[0][0][C].item()
    logit_d = logit_stack[0][0][D].item()

    maximum=max(logit_a, logit_b, logit_c, logit_d)

    if maximum==logit_a:
      final_output = "A"
    elif maximum==logit_b:
      final_output = "B"
    elif maximum==logit_c:
      final_output = "C"
    else:
      final_output = "D"
    return final_output

if __name__ == '__main__':
    question = sys.argv[1].strip()
    option_a = sys.argv[2].strip()
    option_b = sys.argv[3].strip()
    option_c = sys.argv[4].strip()
    option_d = sys.argv[5].strip()

    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,question,option_a,option_b,option_c,option_d)
    print(out)