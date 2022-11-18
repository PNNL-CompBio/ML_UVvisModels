import random 
 
fin = open("esol_training.csv", 'rb') 
f75out = open("85_training_set.csv", 'wb') 
f25out = open("15_testset.csv", 'wb') 
for line in fin: 
    r = random.random() 
    if r < 0.85: 
        f75out.write(line) 
    else: 
        f25out.write(line) 
fin.close() 
f75out.close() 
f25out.close()
