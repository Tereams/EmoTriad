
from random import choice

def show(emos):
    dic={}
    for emo in emos:
        for e in emo[1]:
            if e not in dic:
                dic[e]=1
            else:
                dic[e]+=1
    print(dic)

def stats(emo):
    dic={}
    for e in emo[1]:
        if e not in dic:
            dic[e] = 1
        else:
            dic[e] += 1
    return list(dic.keys())


if __name__ =='__main__':
    f1=open('dialogues_text.txt','r',encoding='utf-8')
    f2=open('dialogues_emotion.txt','r',encoding='utf-8')
    # f3=open('bal_text1.txt','w',encoding='utf-8')
    # f4=open('bal_emo1.txt','w',encoding='utf-8')

    emos=[]
    text=[]
    lines1=f1.readlines()
    lines=f2.readlines()
    for li in zip(lines1,lines):
        line=li[1].strip()
        emo=line.split()
        if '6' not in emo:
            emos.append((li[0],emo))

    show(emos)

    emos2=[]
    for emo in emos:
        a=0
        b=0
        for e in emo[1]:
            if e=='0':
                a+=1
            else:
                b+=1
        if a<=b:
            emos2.append(emo)
    show(emos2)

    emos3=[]
    emos4=[]
    for emo in emos2:
        d=stats(emo)
        if d != ['0','4']:
            emos3.append(emo)
        else:
            emos4.append(emo)
    show(emos3)
    show(emos4)

    emos5=[]
    count=0
    while count<1517:
        e1=choice(emos4)
        for e in e1[1]:
            if e=='4':
                count+=1
        emos5.append(e1)
    show(emos5)
    final_emos=emos3+emos5
    show(final_emos)

    text,emotion=zip(*final_emos)

    # for t in text:
    #     f3.write(t)
    # for e in emotion:
    #     f4.write(' '.join(e)+'\n')

    f1.close()
    f2.close()
    # f3.close()
    # f4.close()

    # emos=[]
    # emos1=[]
    #
    # lines=f2.readlines()
    # for line in lines:
    #     line=line.strip()
    #     emo=line.split()
    #
    #     if '6' not in emo:
    #         for e in emo:
    #             if e !='0' and e!='4':
    #                 emos.append(emo)
    #                 break
    #         else:
    #             emos1.append(emo)
    #
    # dic={}
    # for emo in emos:
    #     for e in emo:
    #         if e not in dic:
    #             dic[e]=1
    #         else:
    #             dic[e]+=1
    # dic1 = {}
    # for emo in emos1:
    #     for e in emo:
    #         if e not in dic1:
    #             dic1[e] = 1
    #         else:
    #             dic1[e] += 1
    # print(dic)
    # print(dic1)
    #
    # dic2 = {}
    # emos2=[]
    # for emo in emos1:
    #     a=0
    #     b=0
    #     for e in emo:
    #         if e=='0':
    #             a+=1
    #         else:
    #             b+=1
    #     if a<=b:
    #         emos2.append(emo)

    # for emo in emos2:
    #     for e in emo:
    #         if e not in dic2:
    #             dic2[e] = 1
    #         else:
    #             dic2[e] += 1
    # print(dic2)