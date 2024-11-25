import re
import os
import torch
import subprocess
from lmdeploy.model import MODELS, BaseChatTemplate
from lmdeploy import ChatTemplateConfig
from utils.conversation import get_conv_template, SeparatorStyle


IMAGE_PLACEHOLDER = '<image>'


def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False


def init_dist(args):
    num_gpus = torch.cuda.device_count()
    args.rank = int(os.getenv('SLURM_PROCID', '0'))
    args.local_rank = args.rank % (num_gpus // args.num_gpus_per_rank)
    args.world_size = int(os.getenv('SLURM_NTASKS', '1'))
    args.local_world_size = num_gpus // args.num_gpus_per_rank

    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(args.local_world_size)

    if 'MASTER_ADDR' not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ['MASTER_ADDR'] = addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '22110'

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=args.rank,
        world_size=args.world_size,
    )
    torch.cuda.set_device(args.local_rank)


def ConvertLMDeployChatTemplate(conv_name):
    template = get_conv_template(conv_name)
    if template.sep_style == SeparatorStyle.BASE:
        json_data = {
            "model_name": "eval_customized_model",
            "system": '',
            "meta_instruction": '',
            "eosys": '',
            "user": template.roles[0],
            "eoh": template.sep,
            "assistant": template.roles[1],
            "eoa": template.sep,
            "separator": "",
            "capability": "chat",
            "stop_words": [template.sep]
            }
    else:
        json_data = {
            "model_name": "eval_customized_model",
            "system": template.system_template.replace('{system_message}', ''),
            "meta_instruction": template.system_message,
            "eosys": template.sep,
            "user": template.roles[0],
            "eoh": template.sep,
            "assistant": template.roles[1],
            "eoa": template.sep,
            "separator": "",
            "capability": "chat",
            "stop_words": [template.sep]
            }
    assert json_data['model_name'] not in MODELS.module_dict.keys()
    MODELS.register_module(json_data['model_name'], module=BaseChatTemplate)
    return ChatTemplateConfig(**json_data)


class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = str(gt_answers[i])
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)
            if has_word(answer, gt_answers):
                return 1
            else:
                return 0
    
    def evaluate_MRR(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        assert type(gt_answers)==list
        for i in range(len(gt_answers)):
            gt_answers[i] = gt_answers[i].replace("\n", " ")
            gt_answers[i] = gt_answers[i].replace("\t", " ")
            gt_answers[i] = gt_answers[i].strip()
            gt_answers[i] = self.processPunctuation(gt_answers[i])
            gt_answers[i] = self.processDigitArticle(gt_answers[i])
            if has_word(answer, gt_answers[i]):
                return 1 / (i + 1)
        return 0.0

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]

        outText = [str(text) for text in outText]
        outText = " ".join(outText)
        return outText


def get_input(item):
    images_list = item['images_list']
    context = item['context']
    question = item['question']
    answer = item['answer']
    meta = item['meta']

    choices = meta['choices']
    choices_image = meta['choices_image_path']

    # context
    num_image_placeholders = context.count(IMAGE_PLACEHOLDER) + question.count(IMAGE_PLACEHOLDER)
    assert num_image_placeholders == len(images_list), f"{context=}\n{num_image_placeholders=}, {len(images_list)=}"
    assert choices is None or choices_image is None

    # retrieval-image
    if choices:
        for c_idx, c in enumerate(choices):
            question = f"{question}\n{chr(c_idx + ord('A'))}. {c}"
        question += "\nAnswer with the option's letter from the given choices directly."

    # reasoning-image
    elif choices_image:
        for c_idx, c in enumerate(choices_image):
            question = f"{question}\n{chr(c_idx + ord('A'))}. <image>"
            images_list.append(c)
        question += "\nAnswer with the option's letter from the given choices directly."

    # counting-image/text
    elif isinstance(answer, list):
        # InternVL-1.5 performs better with this prompt
        question += '\nAnswer the question using a single word or phrase.'

    # retrieval/reasoning-text
    else:
        question += '\nAnswer the question using a single word or phrase.'

    return context, images_list, question, answer
