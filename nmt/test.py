import re
text = "你说：“我爱你。”但是，我并不爱你。所以，你很生气！我不生气吗？“行！”你很生气。"
text = re.sub('([。！？\?])([^’”])',r'\1\n\2',text)#普通断句符号且后面没有引号
text = re.sub('(\.{6})([^’”])',r'\1\n\2',text)#英文省略号且后面没有引号
text = re.sub('(\…{2})([^’”])',r'\1\n\2',text)#中文省略号且后面没有引号
text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',r'\1\n\2',text)#断句号+引号且后面没有引号
text = text.rstrip()    # 去掉段尾的\n，然后
print(text)
# a = re.split(r"[“.*”|。|！|？]", str)
# print(a)