import os
work_path = "/data/speech_data/preprocessed_data/aishell3/"
train_path = os.path.join(work_path,"train_origin.txt")
val_path = os.path.join(work_path,"val_origin.txt")
val_path_write = os.path.join(work_path,"val.txt")
train_path_write = os.path.join(work_path,"train.txt")
id_of_dataset = "2"
FINALS = [
    'a', 'ai', 'ao', 'an', 'ang', 'e', 'er', 'ei', 'en', 'eng', 'o', 'ou',
    'ong', 'ii', 'iii', 'i', 'ia', 'iao', 'ian', 'iang', 'ie', 'io', 'iou',
    'iong', 'in', 'ing', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uo', 'uen',
    'ueng', 'v', 've', 'van', 'vn'
]
INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x', 'y', 'w'
]

lines = []
# if it's combined
for i, path in enumerate([val_path, train_path]):
    with open(path,"r",encoding="utf-8") as f:
        lines.append(f.readlines())

for i, path in enumerate([val_path_write, train_path_write]):
    newline = []
    with open(path, "w", encoding = "utf-8") as f:
        for line in lines[i]:
            line = line.strip("\n")
            line = line.split("|")
            l2s = line[2].strip("{").strip("}") # pinyin 
            l3s = line[3] # tone
            l2s = l2s.split(" ")
            l3s = l3s.split(" ")
            count = 0
            for l2 in l2s:
                # print("-{}-".format(l2))
                if l2[-1].isdigit():
                        count += 1
            # print(count)
            if count != 0:
                print("That's not right! It's combined! shi man!")
                # print(line)
            assert count == 0
            # print(l2s)
            # print(l3s)
            pinyin_tones = []
            raw_pinyins = []
            for i, l2 in enumerate(l2s):
                pinyin_tone = l2s[i]
                if str(l3s[i]) != '0':
                    pinyin_tone = pinyin_tone + str(l3s[i])
                pinyin_tones.append(pinyin_tone)
            #  0 for en 1 for cn
            for i,p in enumerate(pinyin_tones):
                if pinyin_tones[i-1] in INITIALS:
                    raw_pinyins.append(pinyin_tones[i-1]+pinyin_tones[i])
                    continue
                elif pinyin_tones[i] in INITIALS:
                    continue
                else:
                    raw_pinyins.append(pinyin_tones[i])   


            # control the path
            newline = (line[0]+"|"+line[1]+"|"+ "{" + " ".join(pinyin_tones) + "}" + "|" + " ".join(raw_pinyins) +"|"+id_of_dataset)
            # print(newline)
            f.write(newline+"\n")
        