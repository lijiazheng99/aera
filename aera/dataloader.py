import torch
import pandas as pd
import numpy as np
from argparse import Namespace
from torch.utils.data import Dataset
from typing import Any, Union
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CustomDataset2(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        source = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        source['labels'] = self.labels[idx]
        return source

    def __len__(self):
        return len(self.labels)

class PromptGenerator:
    def __init__(self, version):
        if version not in [ 'qkr', '0310', '0314', '0405', '0506', '0621']:
            raise print(f'version not found {version}')
        self.version = version
        if version == '0405':
            self.prompt_base = {}
            for each in ['1', '2', '5', '6']:
                loading_path = f'./dataset/chatgpt_api_0405/prompts/asap-{each}/content.txt'
                with open(loading_path, 'r') as file:
                    content = file.read()
                    self.prompt_base[each] = content

    def generate(self, question, student_answer):
        if question == 1:
            if self.version == 'qkr':
                return  f" Question: A group of students wrote the following procedure for their investigation. Procedure: 1.Determine the mass of four different samples. 2.Pour vinegar in each of four separate, but identical, containers. 3.Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container.  4.After 24 hours, remove the samples from the containers and rinse each sample with distilled water. 5.Allow the samples to sit and dry for 30 minutes. 6.Determine the mass of each sample. The students's data are recorded in the table below. A table contains four columns: Sample, Starting Mass (g), Ending Mass (g), Difference in Mass (g). The sample for the first row is Marble, with 9.8 Starting Mass, 9.4 Ending Mass and -0.4 for Difference in Mass. The sample for the second row is Limestone, with 10.4 Starting Mass, 9.1 Ending Mass and -1.3 for Difference in Mass. The sample for the third row is Wood, with 11.2 Starting Mass, 11.2 Ending Mass and 0.0 for Difference in Mass.  The sample for last row is Plastic, with 7.2 Starting Mass, 7.1 Ending Mass and -0.1 for Difference in Mass.  After reading the group's procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.\n Possible Correct Responses: Needed Information: You need to know how much vinegar was used in each container. You need to know what type of vinegar was used in each container. You need to know what materials to test. You need to know what size/surface area of materials should be used. You need to know how long each sample was rinsed in distilled water. You need to know what drying method to use. You need to know what size/type of container to use. Other acceptable responses.\n Rubric: Score 3: The response describes three additional pieces of information that would be needed to accurately replicate the experiment. Score 2:The response describes two additional pieces of information that would be needed to accurately replicate the experiment. Score 1: The response describes one additional piece of information that would be needed to accurately replicate the experiment. Score 0: The response describes little or no accurate or relevant information from the acid rain investigation.\n {student_answer} "
            elif self.version == '0310':
                return  f" Question: A group of students wrote the following procedure for their investigation. Procedure: 1.Determine the mass of four different samples. 2.Pour vinegar in each of four separate, but identical, containers. 3.Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container.  4.After 24 hours, remove the samples from the containers and rinse each sample with distilled water. 5.Allow the samples to sit and dry for 30 minutes. 6.Determine the mass of each sample. The students's data are recorded in the table below. A table contains four columns: Sample, Starting Mass (g), Ending Mass (g), Difference in Mass (g). The sample for the first row is Marble, with 9.8 Starting Mass, 9.4 Ending Mass and -0.4 for Difference in Mass. The sample for the second row is Limestone, with 10.4 Starting Mass, 9.1 Ending Mass and -1.3 for Difference in Mass. The sample for the third row is Wood, with 11.2 Starting Mass, 11.2 Ending Mass and 0.0 for Difference in Mass.  The sample for last row is Plastic, with 7.2 Starting Mass, 7.1 Ending Mass and -0.1 for Difference in Mass.  After reading the group's procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information.\n Possible Correct Responses: Needed Information: You need to know how much vinegar was used in each container. You need to know what type of vinegar was used in each container. You need to know what materials to test. You need to know what size/surface area of materials should be used. You need to know how long each sample was rinsed in distilled water. You need to know what drying method to use. You need to know what size/type of container to use. Other acceptable responses.\n Rubric: Score 3: The response describes three additional pieces of information that would be needed to accurately replicate the experiment. Score 2:The response describes two additional pieces of information that would be needed to accurately replicate the experiment. Score 1: The response describes one additional piece of information that would be needed to accurately replicate the experiment. Score 0: The response describes little or no accurate or relevant information from the acid rain investigation.\n Student answer:{student_answer}\n What score should this Student answer get and why?"
            elif self.version == '0314':
                return  f" [Question]: A group of students wrote the following procedure for their investigation.\nProcedure:\n1.Determine the mass of four different samples.\n2.Pour vinegar in each of four separate, but identical, containers.\n3.Place a sample of one material into one container and label.\nRepeat with remaining samples, placing a single sample into a single container.\n4.After 24 hours, remove the samples from the containers and rinse each sample with distilled water.\n5.Allow the samples to sit and dry for 30 minutes.\n6.Determine the mass of each sample.\nThe students's data are recorded in the table below.\nA table contains four columns: Sample, Starting Mass (g), Ending Mass (g), Difference in Mass (g).\nThe sample for the first row is Marble, with 9.8 Starting Mass, 9.4 Ending Mass and -0.4 for Difference in Mass.\nThe sample for the second row is Limestone, with 10.4 Starting Mass, 9.1 Ending Mass and -1.3 for Difference in Mass.\nThe sample for the third row is Wood, with 11.2 Starting Mass, 11.2 Ending Mass and 0.0 for Difference in Mass.\nThe sample for last row is Plastic, with 7.2 Starting Mass, 7.1 Ending Mass and -0.1 for Difference in Mass.\nAfter reading the group's procedure, describe what additional information you would need in order to replicate the experiment.\nMake sure to include at least three pieces of information.\n\n[Possible Correct Responses]:\nNeeded Information:\nYou need to know how much vinegar was used in each container.\nYou need to know what type of vinegar was used in each container.\nYou need to know what materials to test.\nYou need to know what size/surface area of materials should be used.\nYou need to know how long each sample was rinsed in distilled water.\nYou need to know what drying method to use.\nYou need to know what size/type of container to use.\nOther acceptable responses.\n\n[Marking Rubric]:\nThe response describes three additional pieces of information that would be needed to accurately replicate the experiment -- 3 points;\nThe response describes two additional pieces of information that would be needed to accurately replicate the experiment -- 2 points;\nThe response describes one additional piece of information that would be needed to accurately replicate the experiment -- 1 point;\nThe response describes little or no accurate or relevant information from the acid rain investigation. -- 0 point.\n\n[Student Answer]:\n{student_answer}\n\nCarefully read the [Question], [Possible Correct Responses], and [Marking Rubric], then compare [Student Answer] with the [Possible Correct Responses], apply the [Marking Rubric] to derive the student score. Please be certain to spell out your reasoning so anyone can verify them. Spell out the [Possible Correct Responses] that the [Student Answer] matches, and also spell out which rule in the [Marking Rubric] is applied."
            elif self.version == '0405':
                return f"{self.prompt_base['1']}\n\n[Student Answer]: {student_answer}\n[Score and Rationale]:"
            else:
                print("Not valid prompt version")
        elif question == 2:
            if self.version == 'qkr':
                return  f" Question: A student performed the following investigation to test four different polymer plastics for stretchability. Procedure: 1. Take a sample of one type of plastic, and measure its length. 2. Tape the top edge of the plastic sample to a table so that it is hanging freely down the side of the table. 3. Attach a clamp to the bottom edge of the plastic sample. 4. Add weights to the clamp and allow them to hang for five minutes. 5. Remove the weights and clamp, and measure the length of the plastic types. 6. Repeat the procedure exactly for the remaining three plastic samples. 7. Perform a second trial (T2) exactly like the first trial (T1). The student recorded the following data from the investigation. The table shows the amount of stretch (in millimeters) for four different types of plastic, labeled as A, B, C, and D, when subjected to two different stretching forces, labeled as T1 and T2. For plastic type A, it stretched 10mm under T1 and 12mm under T2. For plastic type B, it stretched 22mm under T1 and 23mm under T2. For plastic type C, it stretched 14mm under T1 and 13mm under T2. Lastly, for plastic type D, it stretched 20mm under both T1 and T2. a.  Draw a conclusion based on the student’s data. b.   Describe two ways the student could have improved the experimental design and/or validity of the results.\n Sample Response: Conclusions: Plastic sample B has more stretchability than the other polymer plastics. Plastic sample A has the least amount of stretchability compared to the other polymer plastics. Not all polymer plastics have the same stretchability. Different polymer plastics have different stretchability (and are therefore suited for different applications). A reasonable conclusion cannot be drawn due to procedural errors. Other reasonable conclusions Experimental Design Improvements: Provide the before and after measurements for length (Did the samples all start out the same size?). Make sure the samples are all of the same thickness. Variations in thickness could have caused variations in stretchability. Perform additional trials. Some of the samples have similar stretchability (A and C, B and D). Two trials may not be enough to conclusively state that one is more stretchable than the other. Indicate how many weights were added to the clamps (Was it the same number for each sample?). Other acceptable responses\n 3-Point Rubric: Score 3: The response draws a valid conclusion supported by the student’s data and describes two ways the student could have improved the experimental design and/or the validity of the results. Score 2: The response draws a valid conclusion supported by the student’s data and describes one way the student could have improved the experimental design and/or the validity of the results. -or- The response describes two ways the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student’s data. Score 1: The response draws a valid conclusion supported by the student’s data but fails to describe, or incorrectly describes, how the student could have improved the experimental design and/or the validity of the results. -or- The response describes one way the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student's data. Score 0: The response provides little or no correct information from the polymer investigation.\n {student_answer} "
            elif self.version == '0310':
                return  f" Question: A student performed the following investigation to test four different polymer plastics for stretchability. Procedure: 1. Take a sample of one type of plastic, and measure its length. 2. Tape the top edge of the plastic sample to a table so that it is hanging freely down the side of the table. 3. Attach a clamp to the bottom edge of the plastic sample. 4. Add weights to the clamp and allow them to hang for five minutes. 5. Remove the weights and clamp, and measure the length of the plastic types. 6. Repeat the procedure exactly for the remaining three plastic samples. 7. Perform a second trial (T2) exactly like the first trial (T1). The student recorded the following data from the investigation. The table shows the amount of stretch (in millimeters) for four different types of plastic, labeled as A, B, C, and D, when subjected to two different stretching forces, labeled as T1 and T2. For plastic type A, it stretched 10mm under T1 and 12mm under T2. For plastic type B, it stretched 22mm under T1 and 23mm under T2. For plastic type C, it stretched 14mm under T1 and 13mm under T2. Lastly, for plastic type D, it stretched 20mm under both T1 and T2. a.  Draw a conclusion based on the student’s data. b.   Describe two ways the student could have improved the experimental design and/or validity of the results.\n Sample Response: Conclusions: Plastic sample B has more stretchability than the other polymer plastics. Plastic sample A has the least amount of stretchability compared to the other polymer plastics. Not all polymer plastics have the same stretchability. Different polymer plastics have different stretchability (and are therefore suited for different applications). A reasonable conclusion cannot be drawn due to procedural errors. Other reasonable conclusions Experimental Design Improvements: Provide the before and after measurements for length (Did the samples all start out the same size?). Make sure the samples are all of the same thickness. Variations in thickness could have caused variations in stretchability. Perform additional trials. Some of the samples have similar stretchability (A and C, B and D). Two trials may not be enough to conclusively state that one is more stretchable than the other. Indicate how many weights were added to the clamps (Was it the same number for each sample?). Other acceptable responses\n 3-Point Rubric: Score 3: The response draws a valid conclusion supported by the student’s data and describes two ways the student could have improved the experimental design and/or the validity of the results. Score 2: The response draws a valid conclusion supported by the student’s data and describes one way the student could have improved the experimental design and/or the validity of the results. -or- The response describes two ways the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student’s data. Score 1: The response draws a valid conclusion supported by the student’s data but fails to describe, or incorrectly describes, how the student could have improved the experimental design and/or the validity of the results. -or- The response describes one way the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student's data. Score 0: The response provides little or no correct information from the polymer investigation.\n Student answer:{student_answer} \nWhat score should this Student answer get and why?"
            elif self.version == '0314':
                return  f" [Question]: A student performed the following investigation to test four different polymer plastics for stretchability.\nProcedure:\n1. Take a sample of one type of plastic, and measure its length.\n2. Tape the top edge of the plastic sample to a table so that it is hanging freely down the side of the table.\n3. Attach a clamp to the bottom edge of the plastic sample.\n4. Add weights to the clamp and allow them to hang for five minutes.\n5. Remove the weights and clamp, and measure the length of the plastic types.\n6. Repeat the procedure exactly for the remaining three plastic samples.\n7. Perform a second trial (T2) exactly like the first trial (T1).\nThe student recorded the following data from the investigation.\nThe table shows the amount of stretch (in millimeters) for four different types of plastic, labeled as A, B, C, and D, when subjected to two different stretching forces, labeled as T1 and T2.\nFor plastic type A, it stretched 10mm under T1 and 12mm under T2.\nFor plastic type B, it stretched 22mm under T1 and 23mm under T2.\nFor plastic type C, it stretched 14mm under T1 and 13mm under T2.\nLastly, for plastic type D, it stretched 20mm under both T1 and T2.\na. Draw a conclusion based on the student’s data.\nb. Describe two ways the student could have improved the experimental design and/or validity of the results.\n\n[Sample Response]:\nConclusions:\nPlastic sample B has more stretchability than the other polymer plastics.\nPlastic sample A has the least amount of stretchability compared to the other polymer plastics.\nNot all polymer plastics have the same stretchability.\nDifferent polymer plastics have different stretchability (and are therefore suited for different applications).\nA reasonable conclusion cannot be drawn due to procedural errors.\nOther reasonable conclusions Experimental Design Improvements:\nProvide the before and after measurements for length (Did the samples all start out the same size?).\nMake sure the samples are all of the same thickness.\nVariations in thickness could have caused variations in stretchability.\nPerform additional trials.\nSome of the samples have similar stretchability (A and C, B and D).\nTwo trials may not be enough to conclusively state that one is more stretchable than the other.\nIndicate how many weights were added to the clamps (Was it the same number for each sample?).\nOther acceptable responses\n[Marking Rubric]:\nThe response draws a valid conclusion supported by the student’s data and describes two ways the student could have improved the experimental design and/or the validity of the results -- 3 points;\nThe response draws a valid conclusion supported by the student’s data and describes one way the student could have improved the experimental design and/or the validity of the results. -or- The response describes two ways the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student’s data -- 2 points;\nThe response draws a valid conclusion supported by the student’s data but fails to describe, or incorrectly describes, how the student could have improved the experimental design and/or the validity of the results. -or- The response describes one way the student could have improved the experimental design and/or the validity of the results but fails to draw or incorrectly draws a conclusion from the student's data. -- 1 point;\nThe response provides little or no correct information from the polymer investigation. -- 0 point\n\n[Student Answer]:\n{student_answer}\n\nCarefully read the [Question], [Sample Response], and [Marking Rubric], then compare [Student Answer] with the [Sample Response], apply the [Marking Rubric] to derive the student score. Please be certain to spell out your reasoning so anyone can verify them. Spell out the [Sample Response] that the [Student Answer] matches, and also spell out which rule in the [Marking Rubric] is applied."
            elif self.version == '0405':
                return f"{self.prompt_base['2']}\n\n[Student Answer]: {student_answer}\n[Score and Rationale]:"
            else:
                print("Not valid prompt version")
        elif question == 5:
            if self.version == 'qkr':
                return f' Question: Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.\n Key elements: mRNA exits nucleus via nuclear pore.  mRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum.  mRNA bases are read in triplets called codons (by rRNA).  tRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA.  The corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids.  A new corresponding amino acid is added to the tRNA.  Amino acids are linked together to make a protein beginning with a START codon in the P site (initiation).  Amino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination).\n Rubric:  3 points Four key elements, 2 points: Three key elements, 1 point:One or two key elements, 0 points:Other\n {student_answer} '
            elif self.version == '0310':
                return f' Question: Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.\n Key elements: mRNA exits nucleus via nuclear pore.  mRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum.  mRNA bases are read in triplets called codons (by rRNA).  tRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA.  The corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids.  A new corresponding amino acid is added to the tRNA.  Amino acids are linked together to make a protein beginning with a START codon in the P site (initiation).  Amino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination).\n Rubric:  3 points Four key elements, 2 points: Three key elements, 1 point:One or two key elements, 0 points:Other\n Student answer:{student_answer}\n What score should this Student answer get and why?'
            elif self.version == '0314':
                return f' [Question]: Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.\n\n[Key Elements]:\nmRNA exits nucleus via nuclear pore.\nmRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum.\nmRNA bases are read in triplets called codons (by rRNA).\ntRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA.\nThe corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids.\nA new corresponding amino acid is added to the tRNA.\nAmino acids are linked together to make a protein beginning with a START codon in the P site (initiation).\nAmino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination).\n[Marking Rubric]:\nFour key elements -- 3points;\nThree key elements -- 2 points;\nOne or two key elements -- 1 point;\nOther -- 0 points.\n\n[Student Answer]:\n{student_answer}\n\nCarefully read the [Question], [Key Elements], and [Marking Rubric], then compare [Student Answer] with the [Key Elements], apply the [Marking Rubric] to derive the student score. Please be certain to spell out your reasoning so anyone can verify them. Spell out the [Key Elements] that the [Student Answer] matches, and also spell out which rule in the [Marking Rubric] is applied.'
            elif self.version == '0405':
                return f"{self.prompt_base['5']}\n\n[Student Answer]: {student_answer}\n[Score and Rationale]:"
            else:
                print("Not valid prompt version")
        elif question == 6:
            if self.version == 'qkr':
                return f' Question: List and describe three processes used by cells to control the movement of substances across the cell membrane.\n Key elements: Selective permeability is used by the cell membrane to allow certain substances to move across.  Passive transport occurs when substances move from an area of higher concentration to an area of lower concentration.  Osmosis is the diffusion of water across the cell membrane.  Facilitated diffusion occurs when the membrane controls the pathway for a particle to enter or leave a cell.  Active transport occurs when a cell uses energy to move a substance across the cell membrane, and/or a substance moves from an area of low to high concentration, or against the concentration gradient.  Pumps are used to move charged particles like sodium and potassium ions through membranes using energy and carrier proteins.  Membrane-assisted transport occurs when the membrane of the vesicle fuses with the cell membrane forcing large molecules out of the cell as in exocytosis.  Membrane-assisted transport occurs when molecules are engulfed by the cell membrane as in endocytosis.  Membrane-assisted transport occurs when vesicles are formed around large molecules as in phagocytosis.  Membrane-assisted transport occurs when vesicles are formed around liquid droplets as in pinocytosis.  Protein channels or channel proteins allow for the movement of specific molecules or substances into or out of the cell.\n Rubric: 3 points Three key elements, 2 points Two key elements, 1 point One key element, 0 points Other\n {student_answer}'
            elif self.version == '0310':
                return f' Question: List and describe three processes used by cells to control the movement of substances across the cell membrane.\n Key elements: Selective permeability is used by the cell membrane to allow certain substances to move across.  Passive transport occurs when substances move from an area of higher concentration to an area of lower concentration.  Osmosis is the diffusion of water across the cell membrane.  Facilitated diffusion occurs when the membrane controls the pathway for a particle to enter or leave a cell.  Active transport occurs when a cell uses energy to move a substance across the cell membrane, and/or a substance moves from an area of low to high concentration, or against the concentration gradient.  Pumps are used to move charged particles like sodium and potassium ions through membranes using energy and carrier proteins.  Membrane-assisted transport occurs when the membrane of the vesicle fuses with the cell membrane forcing large molecules out of the cell as in exocytosis.  Membrane-assisted transport occurs when molecules are engulfed by the cell membrane as in endocytosis.  Membrane-assisted transport occurs when vesicles are formed around large molecules as in phagocytosis.  Membrane-assisted transport occurs when vesicles are formed around liquid droplets as in pinocytosis.  Protein channels or channel proteins allow for the movement of specific molecules or substances into or out of the cell.\n Rubric: 3 points Three key elements, 2 points Two key elements, 1 point One key element, 0 points Other\n Student answer:{student_answer}\n What score should this Student answer get and why?'
            elif self.version == '0314':
                return f' [Question]: List and describe three processes used by cells to control the movement of substances across the cell membrane.\n\n[Key elements]:\nSelective permeability is used by the cell membrane to allow certain substances to move across.\nPassive transport occurs when substances move from an area of higher concentration to an area of lower concentration.\nOsmosis is the diffusion of water across the cell membrane.\nFacilitated diffusion occurs when the membrane controls the pathway for a particle to enter or leave a cell.\nActive transport occurs when a cell uses energy to move a substance across the cell membrane, and/or a substance moves from an area of low to high concentration, or against the concentration gradient.\nPumps are used to move charged particles like sodium and potassium ions through membranes using energy and carrier proteins.\nMembrane-assisted transport occurs when the membrane of the vesicle fuses with the cell membrane forcing large molecules out of the cell as in exocytosis.\nMembrane-assisted transport occurs when molecules are engulfed by the cell membrane as in endocytosis.\nMembrane-assisted transport occurs when vesicles are formed around large molecules as in phagocytosis.\nMembrane-assisted transport occurs when vesicles are formed around liquid droplets as in pinocytosis.\nProtein channels or channel proteins allow for the movement of specific molecules or substances into or out of the cell.\n[Marking Rubric]:\nThree key elements -- 3points;\nTwo key elements -- 2 points;\nOne key element -- 1 point;\nOther -- 0 points.\n\n[Student Answer]:\n{student_answer}\n\nCarefully read the [Question], [Key Elements], and [Marking Rubric], then compare [Student Answer] with the [Key Elements], apply the [Marking Rubric] to derive the student score. Please be certain to spell out your reasoning so anyone can verify them. Spell out the [Key Elements] that the [Student Answer] matches, and also spell out which rule in the [Marking Rubric] is applied.'
            elif self.version == '0405':
                return f"{self.prompt_base['6']}\n\n[Student Answer]: {student_answer}\n[Score and Rationale]:"
            else:
                print("Not valid prompt version")
        else:
            print("Not valid question number")

def load_data(dataset_name:str="asap", random_seed:int=0, model_config:str="bert-base-uncased"):

    if 'asap' in dataset_name:
        # select category
        num = int(dataset_name.split('-')[1])
        print(num)
        # load asap dataset
        df_train = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/train.csv")
        df_dev = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/val.csv")
        df_test = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/test.csv")

        df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        input_col = 'EssayText'
        label_col = 'Score1'
        label_list = df_train[label_col].unique()
        print(f'Label list:{label_list}')
        num_labels = len(label_list)
    elif 'qkr' in dataset_name:
        num = int(dataset_name.split('-')[1])
        print(num)
        # load asap dataset
        df_train = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/train.csv")
        df_dev = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/val.csv")
        df_test = pd.read_csv(f"./dataset/asap-sas-splitted/asap-{num}/test.csv")

        df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        prompt_generator = PromptGenerator(version='qkr')

        df_train['fullText'] = df_train.apply(lambda x: prompt_generator.generate(x['EssaySet'], x['EssayText']), axis=1)
        df_dev['fullText'] = df_dev.apply(lambda x: prompt_generator.generate(x['EssaySet'], x['EssayText']), axis=1)
        df_test['fullText'] = df_test.apply(lambda x: prompt_generator.generate(x['EssaySet'], x['EssayText']), axis=1)

        input_col = 'fullText'
        label_col = 'Score1'
        label_list = df_train[label_col].unique()
        print(f'Label list:{label_list}')
        num_labels = len(label_list)

    elif 'rationale' in dataset_name:
        num = int(dataset_name.split('-')[1])
        df_train = pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/train_2.jsonl", lines=True) 
        df_dev = pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/val.jsonl", lines=True)
        df_test = pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/test.jsonl", lines=True)

        import re
        regex = r"\d+ point[s]?|No point"       

        def get_score(text):
            match = re.search(regex, text)
            if match:
                if match.group(0)[0] in ['0','1','2','3']:
                    return int(match.group(0)[0])
                else:
                    # score not in [0,1,2,3]
                    return -1
            else:
                # No number matached
                return -1

        def split_score_rationale(df):
            scores = [get_score(each.split(';')[0]) for each in df["gpt-3.5-turbo_top1_content"]]
            rationales = [each.split(';')[1] for each in df["gpt-3.5-turbo_top1_content"]]
            df["Score"] = scores
            df["Rationale"] = rationales
            return df
        
        df_train = split_score_rationale(df_train)
        df_dev = split_score_rationale(df_dev)
        df_test = split_score_rationale(df_test)

        input_col = 'Rationale'
        label_list = df_train['Score'].unique()
        print(f'Label list:{label_list}')
        num_labels = len(label_list)
        label_col = 'Score'

    id2label = {str(i):label for i, label in enumerate(label_list)}
    label2id = {label:str(i) for i, label in enumerate(label_list)}
    dataset_args = Namespace(num_labels=num_labels,labels=label_list, label_col=label_col, id2label=id2label,label2id=label2id, df_test=df_test)

    tokenizer = AutoTokenizer.from_pretrained(model_config)
    print((df_train.shape, df_dev.shape, df_test.shape))
   
    train_encodings = tokenizer(df_train[input_col].values.tolist(),truncation=True)
    dev_encodings = tokenizer(df_dev[input_col].values.tolist(),truncation=True)
    test_encodings = tokenizer(df_test[input_col].values.tolist(),truncation=True)
    
    train_labels = df_train[label_col].values.tolist()
    dev_labels = df_dev[label_col].values.tolist()
    test_labels = df_test[label_col].values.tolist() 

    train_dataset = CustomDataset(train_encodings, train_labels)
    dev_dataset = CustomDataset(dev_encodings, dev_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    return train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args

def load_data_generation(dataset_name:str="asap", random_seed:int=0, model_config:str="google/long-t5-tglobal-large"):

    if 'asap' in dataset_name:
        if 'all' in dataset_name.split('-')[1]:
            # asap-all: use all dataset

            nums = [1,2,5,6]

            df_trains = [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/train_2.jsonl", lines=True) for num in nums]
            df_devs = [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/val.jsonl", lines=True) for num in nums]
            df_tests = [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/test.jsonl", lines=True) for num in nums]

            df_train = pd.concat(df_trains)
            df_dev = pd.concat(df_devs)
            df_test = pd.concat(df_tests)

            gold_label_col = "Score1"
            input_col = "EssayText"
            gpt_output_col = "gpt-3.5-turbo_top1_content"
            gpt_output_label = "gpt-3.5-turbo_top1_score"

            df_dev = df_dev[df_dev[gold_label_col]==df_dev[gpt_output_label]]

            print(type(df_test[gold_label_col]))
            print(df_test[gold_label_col].value_counts())

            prompt_generator = PromptGenerator(version='0405')

            df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            dataset_args = Namespace(df_test=df_test)
        
        elif 'leave' in dataset_name.split('-')[0]:
            # asapleave-n: use all dataset except asap-n

            num = int(dataset_name.split('-')[1])

            nums = [each for each in [1,2,5,6] if each != num ]
            df_trains = [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/train_2.jsonl", lines=True) for num in nums]
            df_devs =   [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/val.jsonl", lines=True) for num in nums]
            df_tests =  [pd.read_json(f"./dataset/chatgpt_api_0405/asap-{num}/test.jsonl", lines=True) for num in nums]

            df_train = pd.concat(df_trains)
            df_dev = pd.concat(df_devs)

            df_train = df_train[df_train["Score1"]==df_train['chatgpt_gen_score']]
            df_dev = df_dev[df_dev["Score1"]==df_dev['chatgpt_gen_score']]

            df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            df_dev = df_dev.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            prompt_generator = PromptGenerator(version='0405')
            print(type(df_test['Score1']))
            print(df_test['Score1'].value_counts())
            print(df_train['chatgpt_gen_score'].value_counts())
            label_col = 'gpt-3.5-turbo_top1_content'
            dataset_args = Namespace(df_test=df_test)
        
        else:
            # asap-n: use asap-n dataset
            num = int(dataset_name.split('-')[1])
            print(num)
            # load asap dataset
            df_train = pd.read_json(f"./dataset/chatgpt_api_0405/{dataset_name}/train_2.jsonl", lines=True) 
            df_dev = pd.read_json(f"./dataset/chatgpt_api_0405/{dataset_name}/val.jsonl", lines=True)
            df_test = pd.read_json(f"./dataset/chatgpt_api_0405/{dataset_name}/test.jsonl", lines=True)

            gold_label_col = "Score1"
            input_col = "EssayText"
            gpt_output_col = "gpt-3.5-turbo_top1_content"
            gpt_output_label = "gpt-3.5-turbo_top1_score"
            label_col = gpt_output_col

            # uncomment this line for filtered experiments
            # df_train = df_train[df_train[gold_label_col]==df_train[gpt_output_label]]
            df_dev = df_dev[df_dev[gold_label_col]==df_dev[gpt_output_label]]

            print(type(df_test[gold_label_col]))
            print(df_test[gold_label_col].value_counts())

            prompt_generator = PromptGenerator(version='0405')

            df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            print(df_train[gpt_output_label].value_counts())

            dataset_args = Namespace(df_test=df_test)

    tokenizer = AutoTokenizer.from_pretrained(model_config)
    print((df_train.shape, df_dev.shape, df_test.shape))

    train_encodings = tokenizer([prompt_generator.generate(each["EssaySet"], each["EssayText"]) for each in df_train.iloc], truncation=True)
    dev_encodings = tokenizer([prompt_generator.generate(each["EssaySet"], each["EssayText"]) for each in df_dev.iloc], truncation=True)
    test_encodings = tokenizer([prompt_generator.generate(each["EssaySet"], each["EssayText"]) for each in df_test.iloc], truncation=True)

    with tokenizer.as_target_tokenizer():
        train_labs=df_train[label_col].values.tolist()
        train_labels = tokenizer(train_labs, padding=True, truncation=True)
        dev_labs=df_dev[label_col].values.tolist()
        dev_labels = tokenizer(dev_labs, padding=True, truncation=True)
        test_labs=df_test[label_col].values.tolist()
        test_labels = tokenizer(test_labs, padding=True, truncation=True)

    print(train_encodings[0].tokens)
    print(len(train_labels["input_ids"]))
    train_labels["labels"] = train_labels["input_ids"]
    dev_labels["labels"] = dev_labels["input_ids"]
    test_labels["labels"] = test_labels["input_ids"]
    train_dataset = CustomDataset2(train_encodings, train_labels["labels"])
    dev_dataset = CustomDataset2(dev_encodings, dev_labels["labels"])
    test_dataset = CustomDataset2(test_encodings, test_labels["labels"])

    return train_dataset, dev_dataset, test_dataset, tokenizer, dataset_args
