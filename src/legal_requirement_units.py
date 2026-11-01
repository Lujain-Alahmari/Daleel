"""Source-position preserving units; no question IDs or gold answers."""
import re
DIGITS=str.maketrans('０１２３４５６７８９٠١٢٣٤٥٦٧٨٩','01234567890123456789')
NUMBER=re.compile(r'(?m)^[ \t]*(\d{1,2})\s*[-.،)]\s*')
LETTER=re.compile(r'(?m)^[ \t]*([أابجدهوزحطيa-zA-Z])\s*[-).]\s*')

def split_source_card(card):
    text=card['source_text'];base=card['source_start']
    scan=text.translate(DIGITS)
    numbers=list(NUMBER.finditer(scan)) if card.get('paragraph_number') is None else []
    blocks=[]
    if numbers:
        for i,m in enumerate(numbers):
            end=numbers[i+1].start() if i+1<len(numbers) else len(text)
            blocks.append((m.group(1),m.start(),end))
    else:
        blocks=[('p',0,len(text))]
    result=[]
    for label,start,end in blocks:
        letters=list(LETTER.finditer(scan,start,end))
        if len(letters)>1:
            for i,m in enumerate(letters):
                stop=letters[i+1].start() if i+1<len(letters) else end
                result.append((label+'.'+m.group(1),card['topic_title']+' / '+label+'.'+m.group(1),
                    base+m.start(),base+stop,base+start,base+letters[0].start()))
        else:
            result.append((label,card['topic_title']+' / '+label,base+start,base+end,base+start,base+start))
    return result
