# -*- encoding: utf-8 -*-
import re

blacklist_person_pattern = r'(tôi|mày|tao|nhe|nhá|nhé)'
blacklist_person_obj = re.compile(blacklist_person_pattern, re.I)