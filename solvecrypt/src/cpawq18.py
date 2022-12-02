# %%
import re
from collections import OrderedDict
# %%
f = open("../misc100.txt", encoding="CP932")
data = f.read()
print(data)

# %%
# tips: https://qiita.com/ShinichiIt0/items/419af2f6342bd50923a8
str_crypt = re.sub('[a-z, !]', "", data)
str_unique = "".join(OrderedDict.fromkeys(str_crypt))