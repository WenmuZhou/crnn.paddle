import os
import string
import shutil
from tqdm import tqdm
import random as rnd
from matplotlib import pyplot as plt
from trdg.generators import GeneratorFromStrings


def load_dict(dict_path):
    """Read the dictionnary file and returns all words in it.
    """
    with open(dict_path, "r", encoding="utf8", errors="ignore", ) as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def create_strings_from_dict(length, allow_variable, count, dict_path):
    """
        Create all strings by picking X random word in the dictionnary
    """
    lang_dict = load_dict(dict_path)
    dict_len = len(lang_dict)
    strings = []
    for _ in range(0, count):
        current_string = ""
        for _ in range(0, rnd.randint(1, length) if allow_variable else length):
            current_string += lang_dict[rnd.randrange(dict_len)]
        strings.append(current_string)
    return strings


def create_strings_randomly(count, let, num, sym, lang):
    """
        Create all strings by randomly sampling from a pool of characters.
    """

    # If none specified, use all three
    if True not in (let, num, sym):
        let, num, sym = True, True, True
    pool = ""
    if let:
        pool += string.ascii_letters
        if lang == "cn":
            pool += "".join([chr(i) for i in range(19968, 40908)])  # Unicode range of CHK characters
    if num:
        pool += "0123456789"
    if sym:
        pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

    if lang == "cn":
        min_seq_len = 1
        max_seq_len = 10
    else:
        min_seq_len = 2
        max_seq_len = 10

    strings = []
    for _ in range(0, count):
        seq_len = rnd.randint(min_seq_len, max_seq_len)
        current_string = "".join([rnd.choice(pool) for _ in range(seq_len)])
        strings.append(current_string)
    return strings


# The generators use the same arguments as the CLI, only as parameters
# strs = create_strings_randomly(100, True, True, True, lang='en')
strs = create_strings_from_dict(10, allow_variable=True, count=5000, dict_path='dict.txt')

generator = GeneratorFromStrings(strings=strs, blur=0, random_blur=True, language='cn', skewing_angle=10,
                                 random_skew=True, distorsion_type=3, margins=(0, 0, 0, 0), size=64)

save_dir = r'/home/zj/桌面/ocr_simple_dataset'
img_folder = os.path.join(save_dir, 'val')
if os.path.exists(img_folder):
    shutil.rmtree(img_folder,ignore_errors=True)
os.makedirs(img_folder, exist_ok=True)
pbar = tqdm(total=len(strs))
with open(os.path.join(save_dir, 'val.txt'), mode='w', encoding='utf8') as f:
    i = 0
    for img, text in generator:
        img_path = os.path.join(img_folder,'{}.jpg'.format(i))
        img.save(img_path)
        f.write('{}\t{}\n'.format(img_path,text))
        i+=1
        if i >= len(strs):
            break
        pbar.update(1)
        # print(text)
        # plt.imshow(img)
        # plt.show()
pbar.close()
