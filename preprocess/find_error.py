import os



label_path = "D:\\competion\\detection\\train\\dataset_divided\\train\\labels"
label_list = os.listdir(label_path)
for item in label_list:
    label_file = os.path.join(label_path,item)
    with open(label_file,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(" ")
            if len(line) < 13:
                print(label_file)
