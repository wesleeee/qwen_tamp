import numpy as np
from get_weight import extract_qkv_weights_for_pruning
from get_attention import get_mean_attention_values
from get_tokens import get_input_tokens_per_block
from get_token_source import get_token_sources
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.utils.prune as prune
import gc
import os

def get_sparsity(image_len,token):
    dict={}
    for block in range(len(token)):
        tokens=token[block]
        visual_token=[]
        text_token=[]
        for i in range(len(tokens)):
            if i>=image_len:
                text_token.append(tokens[i])
            else:
                visual_token.append(tokens[i])
        s_l=0
        s_v=0
        s_vl=0
        text_token=np.array(text_token)
        visual_token=np.array(visual_token)
        # 求语言token内相似度
        index=0
        for i in range(len(text_token)):
            for j in range(i+1, len(text_token)):
                index+=1
                cos=np.inner(text_token[i],text_token[j])
                ii=np.sqrt(np.inner(text_token[i],text_token[i]))
                jj=np.sqrt(np.inner(text_token[j],text_token[j]))
                dist=1-cos/(ii*jj)
                s_l+=dist
        s_l=s_l/index
        # 视觉相似度
        index=0
        for i in range(len(visual_token)):
            for j in range(i+1, len(visual_token)):
                index+=1
                cos=np.inner(visual_token[i],visual_token[j])
                ii=np.sqrt(np.inner(visual_token[i],visual_token[i]))
                jj=np.sqrt(np.inner(visual_token[j],visual_token[j]))
                dist=1-cos/(ii*jj)
                s_v+=dist
        s_v=s_v/index
        # 跨模态相似度
        index=0
        for i in range(len(text_token)):
            for j in range(len(visual_token)):
                index+=1
                cos=np.inner(text_token[i],visual_token[j])
                ii=np.sqrt(np.inner(text_token[i],text_token[i]))
                jj=np.sqrt(np.inner(visual_token[j],visual_token[j]))
                dist=1-cos/(ii*jj)
                s_vl+=dist
        s_vl=s_vl/index

        density=(s_l+s_v+s_vl)/3
        dict[block]=1/density
    return dict







def get_activate_tokens(attention_values,token,num_neighbors,sparsity_dict):
    dict={}
    for block in range(len(token)):
        tokens=token[block]
        attention=attention_values[block]
        attn=attention[:,-1]
        mini_len=min(len(tokens),len(attn))
        tokens=tokens[:mini_len]
        attn=attn[:mini_len]
        print("tokens shape is {}".format(tokens.shape))
        print("attention shape is {}".format(attention.shape))
        distance_dict={}
        # 存储距离由小到大的tokens下标
        for i in range(len(tokens)):
            distance=[]
            for j in range(len(tokens)):
                if i==j:
                    continue
                else:
                    cos=np.inner(tokens[i],tokens[j])
                    ii=np.sqrt(np.inner(tokens[i],tokens[i]))
                    jj=np.sqrt(np.inner(tokens[j],tokens[j]))
                    dist=1-cos/(ii*jj)
                    distance.append(dist)
            distance=np.array(distance)
            distance_dict[i]=np.argsort(distance)
        used_dict={}
        new_token=[]
        for i in range(len(tokens)):
            used_dict[i]=0
# 1:in C_prim,0:not in C_prim
        #更新贡献度
        while len(new_token)<len(tokens):
            new_attn = []
            for i in range(len(tokens)):
                a_i = attn[i]
                cnt_neighbor = 0
                for item in distance_dict[i]:
                    if cnt_neighbor == num_neighbors:
                        break
                    else:
                        if used_dict[item] == 1:
                            continue
                        else:
                            cnt_neighbor += 1
                            cos = np.inner(tokens[i], tokens[item])
                            ii = np.sqrt(np.inner(tokens[i], tokens[i]))
                            jj = np.sqrt(np.inner(tokens[item], tokens[item]))
                            dist = 1 - cos / (ii * jj)
                            e = np.exp(-1 * dist)
                            a_i += e * attn[item]
                new_attn.append(a_i)
            attn = np.array(new_attn)
            # 选择贡献度最高的token
            index = np.argmax(attn)
            used_dict[index] = 1
            new_token.append(tokens[index])
            cnt_neighbor = 0
            for item in distance_dict[index]:
                if cnt_neighbor == num_neighbors:
                    break
                else:
                    if used_dict[item] == 1:
                        continue
                    else:
                        cnt_neighbor += 1
                        cos = np.inner(tokens[index], tokens[item])
                        ii = np.sqrt(np.inner(tokens[index], tokens[index]))
                        jj = np.sqrt(np.inner(tokens[item], tokens[item]))
                        dist = 1 - cos / (ii * jj)
                        e = np.exp(-1 * dist)
                        attn[item] -= e * attn[index]

            # 计算是否超过阈值
            s = 1 / sparsity_dict[block]
            Acc = 0
            cnt_a = 0
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    cos = np.inner(tokens[i], tokens[j])
                    ii = np.sqrt(np.inner(tokens[i], tokens[i]))
                    jj = np.sqrt(np.inner(tokens[j], tokens[j]))
                    dist = 1 - cos / (ii * jj)
                    e = np.exp(-1 * dist)
                    Acc += e
                    cnt_a += 1
            Acc = Acc / cnt_a

            cnt_a = 0
            Acc_prim = 0
            for i in range(len(new_token)):
                for j in range(1 + i, len(new_token)):
                    cos = np.inner(new_token[i], new_token[j])
                    ii = np.sqrt(np.inner(new_token[i], new_token[i]))
                    jj = np.sqrt(np.inner(new_token[j], new_token[j]))
                    dist = 1 - cos / (ii * jj)
                    e = np.exp(-1 * dist)
                    Acc_prim += e
                    cnt_a += 1
            Acc_prim = Acc_prim / max(cnt_a,1)#防止除以0

            Acc_pp = 0
            cnt_a = 0
            for i in range(len(tokens)):
                for j in range(len(new_token)):
                    cos = np.inner(tokens[i], new_token[j])
                    ii = np.sqrt(np.inner(tokens[i], tokens[i]))
                    jj = np.sqrt(np.inner(new_token[j], new_token[j]))
                    dist = 1 - cos / (ii * jj)
                    e = np.exp(-1 * dist)
                    Acc_pp += e
                    cnt_a += 1
            Acc_pp = Acc_pp / max(cnt_a,1)#防止除以0
            flag = Acc + Acc_prim - 2 * Acc_pp - 0.1 * np.sqrt(s)
            if flag<0 and len(new_token)>1:
                break

        dict[block]=np.array([new_token])

    return dict


#返回一个输入后得到的乘上激活值后的weight字典
def process_one_input(qkv_weights,prompt,image_path,model_path):

    print("-------------------------------------------------------------------------------------")
    print("getting attention...")

    attention_means = get_mean_attention_values(model_path, image_path, prompt, device="cuda:0")
    print("-------------------------------------------------------------------------------------")
    print("getting tokens...")

    input_tokens = get_input_tokens_per_block(model_path, image_path, prompt, device="cuda:0")
    print("-------------------------------------------------------------------------------------")
    print("getting image length...")

    image_len = get_token_sources(model_path, image_path, prompt, device="cuda:0")
    print("-------------------------------------------------------------------------------------")
    print("getting sparsity...")

    sparsity_dict=get_sparsity(image_len,input_tokens)
    print("-------------------------------------------------------------------------------------")
    print("getting activate tokens...")

    activate_tokens=get_activate_tokens(attention_means,input_tokens,3,sparsity_dict)
    new_qkv_weights={}
    # sum=0
    #
    # for i in range(36):
    #     sum+=sparsity_dict[i]
    # k=average_sparsity*36/sum
    for i in range(36):
        # sparsity=sparsity_dict[i]*k
        q_weights=np.abs(np.array(qkv_weights["Layer_{}".format(i)]["Query Linear Layer"]))
        k_weights=np.abs(np.array(qkv_weights["Layer_{}".format(i)]["Key Linear Layer"]))
        v_weights=np.abs(np.array(qkv_weights["Layer_{}".format(i)]["Value Linear Layer"]))
        new_token=np.array(activate_tokens[i])
        t=np.linalg.norm(new_token,ord=2,axis=0)
        q_weights=q_weights*t
        k_weights=k_weights*t
        v_weights=v_weights*t
        new_qkv_weights["Layer_{}".format(i)]={}
        new_qkv_weights["Layer_{}".format(i)]["Query Linear Layer"]=q_weights
        new_qkv_weights["Layer_{}".format(i)]["Key Linear Layer"] = k_weights
        new_qkv_weights["Layer_{}".format(i)]["Value Linear Layer"] = v_weights

    return new_qkv_weights,sparsity_dict








def process_multiple_input(qkv_weights,prompts,image_paths,model_path):

    nums=len(prompts)
    new_weights={}
    sparsity_dict={}

    for ii in range(len(prompts)):
        print("process input number {} ...".format(ii))
        prompt=prompts[ii]
        image_path=image_paths[ii]
        new_qkv_weights,new_sparsity_dict=process_one_input(qkv_weights, prompt, image_path, model_path)
        if ii==0:
            new_weights=new_qkv_weights
            sparsity_dict=new_sparsity_dict
        else:
            for key in new_sparsity_dict.keys():
                sparsity_dict[key]=sparsity_dict[key]+new_sparsity_dict[key]

            for layers in new_weights.keys():
                for proj in new_weights[layers].keys():
                    new_weights[layers][proj]+=new_qkv_weights[layers][proj]

    for key in sparsity_dict.keys():
        sparsity_dict[key]=sparsity_dict[key]/nums

    for layers in new_weights.keys():
        for proj in new_weights[layers].keys():
            new_weights[layers][proj]=new_weights[layers][proj]/nums
    return new_weights,sparsity_dict



def pruning(prompts,image_paths,model_path,average_sparsity,pruned_model_path):
    print("-------------------------------------------------------------------------------------")
    print("getting weights...")

    qkv_weights = extract_qkv_weights_for_pruning(model_path)

    new_weights,sparsity_dict=process_multiple_input(qkv_weights,prompts,image_paths,model_path)
    sum=0


    masks={}#剪枝掩码

    for i in range(36):
        sum+=sparsity_dict[i]
    k=average_sparsity*36/sum
    for i in range(36):
        sparsity=k*sparsity_dict[i]
        layer_key="Layer_{}".format(i)
        masks[layer_key]={}
        q_weights=new_weights[layer_key]["Query Linear Layer"]
        k_weights=new_weights[layer_key]["Key Linear Layer"]
        v_weights=new_weights[layer_key]["Value Linear Layer"]

        print("pruning q_weights ...")
        num_q=len(q_weights)*len(q_weights[0])
        mask=np.ones(q_weights.shape)
        num_cut=int(num_q*min(sparsity,1))#防止出现剪枝稀疏率大于1的情况
        col=len(q_weights[0])
        q_weights=np.array(q_weights).flatten()
        indexs=np.argsort(q_weights)[::-1]
        for j in range(num_cut):
            index=indexs[j]
            x=index//col
            y=index%col
            mask[x][y]=0
        masks[layer_key]["Query Linear Layer"]=mask

        print("pruning k_weights ...")
        num_k = len(k_weights) * len(k_weights[0])
        mask = np.ones(k_weights.shape)
        num_cut = int(num_k * min(sparsity, 1) ) # 防止出现剪枝稀疏率大于1的情况，向下取整
        col = len(k_weights[0])
        k_weights = np.array(k_weights).flatten()
        indexs = np.argsort(k_weights)[::-1]
        for j in range(num_cut):
            index = indexs[j]
            x = index // col
            y = index % col
            mask[x][y] = 0
        masks[layer_key]["Key Linear Layer"] = mask

        print("pruning v_weights ...")
        num_v = len(v_weights) * len(v_weights[0])
        mask = np.ones(v_weights.shape)
        num_cut = int(num_v * min(sparsity, 1))  # 防止出现剪枝稀疏率大于1的情况
        col = len(v_weights[0])
        v_weights = np.array(v_weights).flatten()
        indexs = np.argsort(v_weights)
        for j in range(num_cut):
            index = indexs[j]
            x = index // col
            y = index % col
            mask[x][y] = 0
        masks[layer_key]["Value Linear Layer"] = mask


        print("---------------------------------------------------------")
        print("applying masks on weights ...")
        qkv_weights[layer_key]["Query Linear Layer"] = masks[layer_key]["Query Linear Layer"]*qkv_weights[layer_key]["Query Linear Layer"]
        qkv_weights[layer_key]["Key Linear Layer"]=masks[layer_key]["Key Linear Layer"]*qkv_weights[layer_key]["Key Linear Layer"]
        qkv_weights[layer_key]["Value Linear Layer"]=masks[layer_key]["Value Linear Layer"]*qkv_weights[layer_key]["Value Linear Layer"]


    print("---------------------------------------------------------------")
    print("writing pruned weights into model ...")
# 1. 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto",
    device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 2. 遍历写回
    for layer_idx, layer in enumerate(model.language_model.layers):
        key = f"Layer_{layer_idx}"
        if key not in qkv_weights:
            continue

        # 取出三条 numpy 权重
        np_q = qkv_weights[key]["Query Linear Layer"]
        np_k = qkv_weights[key]["Key Linear Layer"]
        np_v = qkv_weights[key]["Value Linear Layer"]

        # 转成 Tensor，并放到 model.device
        # 注意：.detach_() 只是把 requires_grad 置为 False，不影响计算图
        with torch.no_grad():
            layer.self_attn.q_proj.weight.data = torch.from_numpy(np_q).to(device)
            layer.self_attn.k_proj.weight.data = torch.from_numpy(np_k).to(device)
            layer.self_attn.v_proj.weight.data = torch.from_numpy(np_v).to(device)

        print(f"Injected Layer {layer_idx}: Q {np_q.shape}, K {np_k.shape}, V {np_v.shape}")

    # 3. 保存到新目录
    os.makedirs(pruned_model_path, exist_ok=True)

    model.save_pretrained(pruned_model_path,safe_serialization=True)
    processor.save_pretrained(pruned_model_path)

    print(f"New model with injected QKV weights saved to: {pruned_model_path}")


if __name__ == "__main__":
    prompts=["tell me about the picture"]
    image_paths=["/home/chuziyuan/zju/qwen_tamp/test_image/蓝桥杯.jpg"]
    pruned_model_path="/home/chuziyuan/zju/qwen_tamp/my_pruned_qwen_model"
    rate=0.3
    model_path="/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-OVD-0321"
    pruning(prompts,image_paths,model_path,rate,pruned_model_path)















