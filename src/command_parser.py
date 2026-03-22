import spacy 
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc
import time

class CommandParser:

    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.doc = None
        self.threshhold = 0.65

        # Priority List for which pattern applies
        self.pattern_priority = {
            "low_level_movement": 0,
            "multiple_objects": 1,
            "object_singular": 2,
            "object_plural": 3
        }

        # Available functions
        self.functions = {"grab_brick", "place_brick", "sort_all_bricks", "get_all_bricks",
                          "get_all_collision_free_bricks", "move_arm", "turn_arm", "put_down_brick"
                        }
        
        # Primitive Vocab Lists
        self.vocab = {
            "brick": ["brick", "block", "cube", "lego", "duplo", "it", "this", "that", "one"],
            "ground": ["ground", "floor", "table", "surface"],

            "grab": ["pick up", "grab", "get", "lift", "bring", "give", "manipulate"],

            "place_near": ["put near", "place next to", "set next to", "lay near"],
            "put_down": ["put down", "place down", "set down", "lay down"],
            "place_on": ["put on", "place on", "set on", "lay on"],

            "sort": ["sort", "organize", "order", "arrange"],
            "display": ["show", "display", "present", "demonstrate"],

            "move": ["move", "go", "travel", "translate", "slide"],
            "turn": ["rotate", "turn", "spin","twist"],

            "direction": ["up","upwards", " upwards",
                          "down", "downwards", "downward",
                          "front", "forwards"," forward",
                          "back", "backwards","backward",
                          "left", "leftwards"," leftward",
                          "right", "rightwards", "rightward"],
            "unit": ["unit", "centimeter", "cm", "degree"],
        }

        self.matcher = self._init_matcher()

    def _split_commands(self, example):
        self.doc = self.nlp(example)
        seperators = {",", ";", "and", "then"}
        chunks = []
        start = 0

        for token in self.doc:
            if token.text in seperators:
                if token.i > start:
                    chunks.append(self.doc[start:token.i])
                start = token.i + 1
        
        if start < len(self.doc):
            chunks.append(self.doc[start:])
        
        return chunks

    # Find inferred command command
    def infer(self, sentence):
        self.doc = self.nlp(sentence)
        chunks = [1]
        result = {
            "command": {
                "function_name": [],
                "arguments": []
            },
            "intent": {
                "object": [],
                "gazeword": []
            }
        }
        matches = self.matcher(self.doc)
        if not matches:
            print("command parser - Error: NLP found no grammar matches")
            result["command"]["function_name"].append("invalid") 
            return result
        matches = self._select_matches(matches)
        for match in matches:
            pattern = match["label"]
            words = [(self.doc[i].text, i) for i in match["token_ids"]]
            print(f"command parser: found grammar match {pattern}")
            print(f"with words {words}")
        for match in matches:
            pattern = match["label"]
            token_ids = match["token_ids"]
            print(f"command parser: found grammar match {pattern}")

            if pattern == "low_level_movement":
                function_name, vector = self._simple_move_calc(token_ids)

                if function_name is None:
                    print("command parser - Error: Not a valid command for low level movement")
                    result["command"]["function_name"].append("invalid")
                    continue
                
                result["command"]["function_name"].append(function_name)
                result["command"]["arguments"].append([vector])
            else:
                function_name = self._vocab_matcher(pattern, token_ids)

                # if intention allignment is required get gazewords
                if function_name == "grab_brick" or function_name == "place_brick":
                    intent = self._get_gazewords(token_ids)
                    result["command"]["arguments"].append("intent")
                    result["intent"]["object"].append(intent["object"])
                    result["intent"]["gazeword"].append(intent["gazeword"])
                result["command"]["function_name"].append(function_name)
        return result
    
    def _select_matches(self, raw_matches):
        matches = []
        #represent matches with dictionary for easy manipulation
        for raw_match in raw_matches: 
            match_id, token_ids = raw_match
            match = {"label": self.nlp.vocab.strings[match_id],
                     "verb_idx": token_ids[0],
                     "token_ids": token_ids}
            matches.append(match)
        
        #sort matches according to priorty list and idx span
        matches.sort(key=lambda m: (self.pattern_priority[m["label"]], self._span(m) ))

        selected = []
        used_tokens = set()
        used_verbs = {}
        for match in matches:

            verb = match["verb_idx"]
            tokens = {idx for idx in match["token_ids"]}

            if verb in used_verbs:
                continue

            if tokens & used_tokens:
                continue
            selected.append(match)

            used_verbs[verb] = match
            used_tokens |= tokens
        
        selected.sort(key=lambda m: m["verb_idx"])
        return selected

    def _span(self, match):
        idxs  = match["token_ids"]
        return max(idxs) - min(idxs)
   
    # Create a dependency matcher with patterns for every command
    def _init_matcher(self):
        # Defined Patterns for Command

        # pattern for verb -> object(singular), relevant for grab_brick

        # pattern for verb -> object(singular) with prp (up or down) attached to verb, relevant for grab_brick and put_down_brick
        object_singular_prp = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "prt",
                "RIGHT_ATTRS": {
                    "DEP": "prt"
                },
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NN", "NNP", "PRP", "DT", "CD"]}
                }
            }
        ]

        # pattern for verb -> objects(plural) relevant for sort_all_bricks and display_all_bricks
        object_plural = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                } 
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NNS", "PRON", "NNPS"]}
                }
            }
        ] 

        # pattern for verb -> object_1 -> prep -> object_2 relevant for place_bricks and put_down_brick
        multiple_objects = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NN", "NNP", "PRP", "DT", "CD"]}
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {
                    "DEP": "prep",
                }
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "target",
                "RIGHT_ATTRS":{
                    "DEP": "pobj",
                    "TAG": {"IN": ["NN","NNP", "PRP", "DT", "CD"]}
                }
            }
        ]

        multiple_objects_alt = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NN", "NNP", "PRP", "DT", "CD"]}
                }
            },
            {
                "LEFT_ID": "object",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {
                    "DEP": "prep",
                }
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "target",
                "RIGHT_ATTRS":{
                    "DEP": "pobj",
                    "TAG": {"IN": ["NN","NNP", "PRP", "DT", "CD"]}
                }
            }
        ]

        # pattern for verb -> object_1 -> prep -> object_2 with amod inbetween relevant for place_bricks
        multiple_objects_amod = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NN", "NNP", "PRP", "DT", "CD"]}
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "advmod",
                "RIGHT_ATTRS": {
                    "DEP": {"IN": ["advmod","amod"]}
                }
            },
            {
                "LEFT_ID": "advmod",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {
                    "DEP": "prep",
                }
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "target",
                "RIGHT_ATTRS":{
                    "DEP": "pobj",
                    "TAG": {"IN": ["NN","NNP", "PRP", "DT", "CD"]}
                }
            }
        ]      

        multiple_objects_amod_alt = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {
                    "DEP": "dobj",
                    "TAG": {"IN": ["NN", "NNP", "PRP", "DT", "CD"]}
                }
            },
            {
                "LEFT_ID": "object",
                "REL_OP": ">",
                "RIGHT_ID": "amod",
                "RIGHT_ATTRS": {
                    "DEP": {"IN": ["advmod","amod"]}
                }
            },
            {
                "LEFT_ID": "amod",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {
                    "DEP": "prep",
                }
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "target",
                "RIGHT_ATTRS":{
                    "DEP": "pobj",
                    "TAG": {"IN": ["NN","NNP", "PRP", "DT", "CD"]}
                }
            }
        ]    

        # simple pattern for low level movements collects verb unit, direction and number
        low_level_movement = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {
                    "POS": "VERB"
                    }
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">>",
                "RIGHT_ID": "unit",
                "RIGHT_ATTRS": {
                    "POS": "NOUN",
                    "LEMMA":{"IN": self.vocab["unit"]}
                }
            },
            {
                "LEFT_ID": "unit",
                "REL_OP": ">",
                "RIGHT_ID": "number",
                "RIGHT_ATTRS": {
                    "DEP": "nummod"
                }
            },            
            {
                "LEFT_ID": "verb",
                "REL_OP": ">>",
                "RIGHT_ID": "direction",
                "RIGHT_ATTRS": {
                    "LEMMA": {"IN": self.vocab["direction"]}
                }
            }

        ]
       
        # Create Matcher and add Commands
        matcher = DependencyMatcher(self.nlp.vocab)
        matcher.add("object_singular", [object_singular_prp])
        matcher.add("object_plural", [object_plural])
        matcher.add("multiple_objects", [multiple_objects, multiple_objects_alt, multiple_objects_amod, multiple_objects_amod_alt])
        matcher.add("low_level_movement", [low_level_movement])

        return matcher
    
    # Find the refered objects and their associated gazewords (intention allignement)
    def _get_gazewords(self,token_ids):

        result = {
            "object": [],
            "gazeword": []
        }

        # Get all objects that refer to bricks with earlier dependency matching
        object_tokens = [self.doc[i] for i in token_ids 
                         if self.doc[i].dep_ in {"pobj", "dobj"}]   

        for token in object_tokens:
            result["object"].append(token.text)

            # Gather componenets of their noun phrases sorted in 3 categories
            determiner = []
            article = []
            adjective = []    
            for child in token.children:
                if child.dep_ == "det" and child.morph.get("PronType") == ['Dem']:
                    determiner.append(child)
                elif child.dep_ == "det" and child.morph.get("PronType") == ['Art']:
                    article.append(child)
                elif child.dep_ == "amod":
                    adjective.append(child)            
            
            # Select gazeword 
            gazeword = None 
            if determiner:
                gazeword = determiner[0].text
            elif adjective:
                gazeword = adjective[0].text   
            elif article:
                gazeword = article[0].text
            else:
                gazeword = token.text     
            result["gazeword"].append(gazeword) 

        return result               
    
    # Calculates a Vector and decides between rotate and move
    def _simple_move_calc(self, token_ids):
        tokens = [self.doc[i] for i in token_ids]
        #vocab = tokens[0].doc.vocab 

        verb =  [t for t in tokens if t.pos_ == "VERB"][0]
        unit = [t for t in tokens if t.lemma_ in self.vocab["unit"]][0]
        direction = [t for t in tokens if t.lemma_ in self.vocab["direction"]][0]   
        number = [t for t in tokens if t.dep_ == "nummod"][0]

        num = int(number.text)
        vector = None

        if direction.lemma_ in ("forwards", "front"):
            vector =[num, 0, 0]
        elif direction.lemma_ in ("backwards", "back"):
            vector = [-num, 0, 0]
        elif direction.lemma_ in ("right", "rightwards"):
            vector = [0, num, 0]
        if direction.lemma_ in ("left", "leftwards"):
            vector = [0, -num, 0]
        elif direction.lemma_ in ("upwards", "up"):
            vector = [0, 0, num]
        elif direction.lemma_ in ("downwards", "down"):
            vector = [0, 0, -num]


        verb_match = self._similarity_check(verb,["move", "turn"])
        print(verb_match)

        if verb_match == "move" and unit.lemma_ in ("centimeter","cm", "unit"):
            return "move_arm", vector

        elif verb_match == "turn" and unit.lemma_ in ("degree", "unit"):

            if direction.lemma_ not in ("left", "leftward", "right", "rightward"):
                print("Invalid Turn Direction")
                return None, None
            
            return "turn_arm", vector
            
        return None, None 

    # Checks vocabulary and determines correct command based of sentence pattern
    def _vocab_matcher(self, pattern, token_ids):
        tokens = [self.doc[i] for i in token_ids]
        vocab = tokens[0].doc.vocab

        match pattern:
            case "object_singular":
                object_token = [t for t in tokens if t.dep_ == "dobj"][0]

                if self._similarity_check(object_token, ["brick"]) is not None:

                    phrase_tokens = [t for t in tokens if t.dep_ != "dobj"]
                    phrase_doc = Doc(vocab, words=[t.text for t in phrase_tokens])

                    closest_match = self._similarity_check(phrase_doc, ["grab", "put_down"])

                    if closest_match == "grab":
                        return "grab_brick"
                    elif closest_match == "put_down":
                        return "put_down_brick"
                
                return "invalid"

            case "object_plural":
                object_token = [t for t in tokens if t.dep_ == "dobj"][0]

                if self._similarity_check(object_token, ["brick"]) is not None:

                    phrase_tokens = [t for t in tokens if t.dep_ != "dobj"]
                    phrase_doc = Doc(vocab, words=[t.text for t in phrase_tokens])

                    closest_match = self._similarity_check(phrase_doc, ["sort", "display"])

                    if closest_match == "sort":
                        return "sort_all_bricks"
                    elif closest_match == "display":
                        context_token_id = token_ids[1]
                        if context_token_id >= 3 and self.doc[context_token_id - 1].text == "free" and self.doc[context_token_id - 2].text == "collision":
                            return "get_all_collision_free_bricks"
                        else:
                            return "get_all_bricks"
                
                return "invalid"
            
            case "multiple_objects":
                object_token = [t for t in tokens if t.dep_ == "dobj"][0]

                if self._similarity_check(object_token, ["brick"]) is not None:

                    pobject_token = [t for t in tokens if t.dep_ == "pobj"][0]

                    phrase_tokens = [t for t in tokens if t.dep_ not in ("dobj", "pobj")]
                    phrase_doc = Doc(vocab, words=[t.text for t in phrase_tokens])

                    closest_object_match = self._similarity_check(pobject_token, ["brick", "ground"])
                    closest_phrase_match = self._similarity_check(phrase_doc, ["place_near", "place_on"])

                    if closest_object_match == "brick" and closest_phrase_match == "place_near":
                        return "place_brick"
                    elif closest_object_match == "ground" and closest_phrase_match == "place_on":
                        return "put_down_brick"
                    
                return "invalid"

        return "invalid"   

    # Calculates similarty of phrase or word to predefined word lists
    def _similarity_check(self, phrase_doc, vocab_keywords):
        vocabs = [self.vocab[keyword] for keyword in vocab_keywords]

        list_similarities = []

        for vocab_list in vocabs:

            candidates = [self.nlp(p) for p in vocab_list]

            sims = [phrase_doc.similarity(cand) for cand in candidates]

            collective_sim = max(sims)

            list_similarities.append(collective_sim)
        
        best_idx = None
        best_sim = self.threshhold
        for idx, sim in enumerate(list_similarities):
            if sim >= best_sim:
                best_sim = sim
                best_idx = idx
        
        if best_idx is not None:
            return vocab_keywords[best_idx]
        else:
            return None
