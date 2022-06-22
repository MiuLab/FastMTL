import sys

IGNORE = ['MNLI-m', 'MNLI-mm', 'RTE', 'QNLI']
POS = ['acceptable', 'positive', 'equivalent', 'duplicate']
NEG = ['unacceptable', 'negative', 'not_equivalent', 'not_duplicate']

if sys.argv[1] not in IGNORE:
    result = []
    with open(sys.argv[1], 'r') as F:
        cnt = 0
        for line in F.readlines():
            line = line.strip()
            split = line.split('\t')
            if "STS-B" in sys.argv[1] and cnt>0:
                if float(split[1])<0:
                    result.append(split[0] + '\t' + '0.000\n')
                elif float(split[1])>5:
                    result.append(split[0] + '\t' + '5.000\n')
                else:
                    result.append(line + '\n')
            elif split[1] in POS:
                result.append(split[0] + '\t' + '1\n')
            elif split[1] in NEG:
                result.append(split[0] + '\t' + '0\n')
            else:
                result.append(line + '\n')
            cnt+=1
    with open(sys.argv[1], 'w') as F:
        for line in result:
            F.write(line)
