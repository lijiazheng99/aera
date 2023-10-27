# Dataset
folders explanation:

```query_chatgpt.py``` is the python script used to query the chatgpt api. Please ensure timeout_decorator, openai and tenacity are installed. You need your own api key to run the script.

```asap-sas-splitted``` contains the specific train/dev/test splits employed in our experiments. The original dataset was sourced from [The Hewlett Foundation: Short Answer Scoring](https://www.kaggle.com/competitions/asap-sas). For our study, the training and development sets were derived from the original training set, with Score1 selected as the ground truth label.

```chatgpt_api_0310``` correspond to the **Simple Instruction** query result from the paper.

```chatgpt_api_0314``` correspond to the **Complex Instruction** query result from the paper.

```chatgpt_api_0405``` correspond to the **Example Instruction** query result from the paper.