def detect(csvfile):
    with open(csvfile, 'r') as mycsvfile:
        header=mycsvfile.readline()
        if header.find(";")!=-1:
            return ";"
        if header.find(",")!=-1:
            return ","
    return ";"