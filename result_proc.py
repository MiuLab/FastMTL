import sys

IGNORE = ['MNLI-m', 'MNLI-mm', 'RTE', 'QNLI']
POS = ['acceptable', 'positive', 'equivalent', 'duplicate']
NEG = ['unacceptable', 'negative', 'not_equivalent', 'not_duplicate']

if sys.argv[1] not in IGNORE:
    result = []
    with open(sys.argv[1], 'r') as F:
        for line in F.readlines():
            line = line.strip()
            split = line.split('\t')
            if split[1] in POS:
                result.append(split[0] + '\t' + '1\n')
            elif split[1] in NEG:
                result.append(split[0] + '\t' + '0\n')
            else:
                result.append(line + '\n')
    with open(sys.argv[1], 'w') as F:
        for line in result:
            F.write(line)

