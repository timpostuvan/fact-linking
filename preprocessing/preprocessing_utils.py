relation_to_natural = {"AtLocation": "$H is located or found at $T",
                       "CapableOf": "$H is capable of $T",
                       "Causes": "$H causes $T",
                       "CausesDesire": "$H makes someone want $T",
                       "CreatedBy": "$H is created by $T",
                       "Desires": "$H desires $T",
                       "HasA": "$H has $T",
                       "HasFirstSubevent": "$H begins with that $T",
                       "HasLastSubevent": "$H ends with that $T",
                       "HasPrerequisite": "$H requires that $T",
                       "HasProperty": "$H can be characterized by having or being $T",
                       "HasSubEvent": "$H includes that $T",
                       "HinderedBy": "$H can be hindered by $T",
                       "InstanceOf": "$H is an example of $T",
                       "isAfter": "$H happens after $T",
                       "isBefore": "$H happens before $T",
                       "MadeOf": "$H is made of $T",
                       "MadeUpOf": "$H is made up of $T",
                       "MotivatedByGoal": "$H is a step towards accomplishing the goal $T",
                       "NotDesires": "$H does not desire $T",
                       "ObjectUse": "$H is used for $T",
                       "UsedFor": "$H is used for $T",
                       "oEffect": "$H, as a result, PersonY will $T",
                       "oReact": "$H, as a result, PersonY feels $T",
                       "oWant": "$H, as a result, PersonY wants $T",
                       "PartOf": "$H is a part of $T",
                       "ReceivesAction": "$H can receive or be affected by the action $T",
                       "xAttr": "$H, so PersonX is seen as $T",
                       "xEffect": "$H, as a result, PersonX will $T",
                       "xIntent": "$H because PersonX wanted $T",
                       "xNeed": "$H, but before, PersonX needed $T",
                       "xReact": "$H, as a result, PersonX feels $T",
                       "xReason": "$H because $T",
                       "xWant": "$H, as a result, PersonX wants $T"}


def convert_fact_to_text(head: str, relation: str, tail: str):
    if relation in relation_to_natural:
        text = relation_to_natural[relation]
        text = text.replace("$H", head)
        text = text.replace("$T", tail)
    elif relation == "isFilledBy":
        text = head.replace("___", tail)
    else:
        raise ValueError(f"Unknown relation: {relation}")
    
    return text
