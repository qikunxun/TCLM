import json

with open('rules.txt') as fd:
    for line in fd:
        if not line: continue
        data = json.loads(line.replace('http://dbpedia.org/ontology/ai4dm/', '').replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', '').replace('http://www.w3.org/2002/07/owl#', ''))
        rules = data['rules']
        weight = data['weight']
        for rule in rules:
            # if 'type(' in rule: continue
            if weight < 0: continue
            atoms = rule.split(' ∧ ')
            print(weight, len(atoms), rule)