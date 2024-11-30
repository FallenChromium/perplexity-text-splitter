from .ppl_caclucate import Chunking
from typing import List, Dict
import math 
from nltk.tokenize import sent_tokenize
import torch
import time

def split_text_by_punctuation(text, language):
    """
    Splits input text into sentences based on language-specific punctuation.
    
    Args:
        text (str): Input text to be split
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        
    Returns:
        list: List of sentences with length constraints applied
    """
    full_segments = sent_tokenize(text, language)
    return [segment.strip() for segment in full_segments]

def find_minima(values, threshold):
    """
    Finds local minima points in a sequence of values.
    
    Args:
        values (list): List of values to analyze
        threshold (float): Threshold for considering a point a minimum
        
    Returns:
        list: Indices of identified minima
    """
    minima_indices = []
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i)
    return minima_indices

def find_minima_dynamic(values, threshold, threshold_zlist):
    """
    Similar to find_minima but with dynamic threshold adjustment.
    
    Args:
        values (list): List of values to analyze
        threshold (float): Threshold for considering a point a minimum
        threshold_zlist (list): List to maintain history of threshold values
        
    Returns:
        tuple:
            - list: Indices of identified minima
            - float: Updated threshold
            - list: Threshold history
    """
    minima_indices = []
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)
                threshold_zlist.append(min(values[i - 1]-values[i], values[i + 1]-values[i]) )
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i)
                threshold_zlist.append(values[i - 1]-values[i])
        if len(threshold_zlist)>=100:
            last_ten = threshold_zlist#[-100:]
            avg=min(last_ten)
            threshold=avg
    return minima_indices, threshold, threshold_zlist

def extract_by_html2text_db_chongdie(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:
    """
    Processes text using perplexity-based chunking with overlap.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        
    Returns:
        list: List of chunked text segments
    """
    cleaned_text=sub_text

    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        # if i==0:
        #     tmp_index.append(0)
        #     tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i],split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks

def extract_by_html2text_db_nolist(sub_text, model, tokenizer, threshold, language='zh') -> List[str]:
    """
    Similar to db_chongdie but without overlap.
    Ensures first sentence is always included in first chunk.
    More suitable for clean text segmentation.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        
    Returns:
        list: List of chunked text segments
    """
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks

def extract_by_html2text_db_dynamic(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh') -> List[str]:
    """
    Implements dynamic threshold adjustment during chunking.
    Returns chunks plus updated threshold information.
    Useful for adaptive text processing.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        threshold_zlist (list): List to maintain history of threshold values
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        
    Returns:
        tuple:
            - list: List of chunked text segments
            - float: Updated threshold
            - list: Threshold history
    """
    temp_para=sub_text 
    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para

    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices,threshold,threshold_zlist=find_minima_dynamic(first_cluster_ppl,threshold,threshold_zlist)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)
    # temp_para经过困惑度分组
    return final_chunks,threshold,threshold_zlist

def extract_by_html2text_db_dynamic_batch(sub_text, model, tokenizer, threshold, threshold_zlist, language='zh', past_key_values=None) -> List[str]:
    """
    Batch processing version of dynamic chunking.
    Processes text in batches of 4096 tokens.
    Handles long sequences efficiently.
    Maintains context using past_key_values.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        threshold_zlist (list): List to maintain history of threshold values
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        past_key_values (tuple): Past key values for maintaining context
        
    Returns:
        tuple:
            - list: List of chunked text segments
            - float: Updated threshold
            - list: Threshold history
    """
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)


    batch_size = 4096   #6000

    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    loss=torch.tensor([], device=model.device,dtype=torch.long)
    for i in range(total_batches): 
        start=i*batch_size
        end=start+batch_size
        input_ids_tmp=input_ids[:,start:end]

        attention_mask_tmp=attention_mask[:,:end]
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp],dim=-1)
        attention_mask_tmp=torch.cat([ attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)  ],dim=-1)
        
        size=input_ids_tmp.shape[1]
        if attention_mask_tmp.shape[1]>24576:  #72000
            past_key_values = [  
                [k[:, :, size+1: ], v[:, :, size+1: ]]  
                for k, v in past_key_values  
            ]
            attention_mask_tmp=attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
            # print('111',attention_mask_tmp.shape,past_key_values[0][0].shape[2])
        
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp],dim=-1)
        # print(input_ids_tmp.shape,attention_mask_tmp.shape,past_key_values[0][0].shape[2],loss.shape)
            
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
            # index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
        index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices,threshold,threshold_zlist=find_minima_dynamic(first_cluster_ppl,threshold,threshold_zlist)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)
    # temp_para经过困惑度分组
    return final_chunks,threshold,threshold_zlist

def extract_by_html2text_db_bench(sub_text, model, tokenizer, threshold, language='zh', batch_size=4096, max_txt_size=9000, past_key_values=None) -> List[str]:
    """
    Benchmark version of chunking algorithm.
    Configurable batch size and max text size.
    Optimized for performance testing.
    Returns clean chunks without overlap.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        batch_size (int): Batch size for processing
        max_txt_size (int): Maximum text size for processing
        past_key_values (tuple): Past key values for maintaining context
        
    Returns:
        list: List of chunked text segments
    """
    temp_para=sub_text
    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para

    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)
  
    batch_size = batch_size

    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    print('111',input_ids.shape[1])
    loss=torch.tensor([], device=model.device,dtype=torch.long)
    for i in range(total_batches): 
        start=i*batch_size
        end=start+batch_size
        input_ids_tmp=input_ids[:,start:end]
        attention_mask_tmp=attention_mask[:,:end]
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp],dim=-1)
        attention_mask_tmp=torch.cat([ attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)  ],dim=-1)
        
        size=input_ids_tmp.shape[1]
        if attention_mask_tmp.shape[1]>max_txt_size:  
            past_key_values = [  
                [k[:, :, size+1: ], v[:, :, size+1: ]]  
                for k, v in past_key_values  
            ]
            attention_mask_tmp=attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
            # print('111',attention_mask_tmp.shape,past_key_values[0][0].shape[2])
        
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp],dim=-1)
        # print(input_ids_tmp.shape,attention_mask_tmp.shape,past_key_values[0][0].shape[2],loss.shape)
            
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
            # index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
        index+=len_sentences[i]
    # print('333',first_cluster_ppl)
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks


def llm_chunker_ppl(sub_text, model, tokenizer, threshold, language='zh', batch_size=4096, max_txt_size=9000, dynamic_merge='no', target_size=200)-> List[str]:
    """
    Main entry point for perplexity-based chunking.
    Handles both short (<4096) and long text.
    Optional dynamic merging of small chunks.
    Language-aware processing (English/Chinese).
    Returns final chunked paragraphs.
    
    Args:
        sub_text (str): Input text to be chunked
        model (torch.nn.Module): Language model
        tokenizer (torch.nn.Module): Tokenizer
        threshold (float): Threshold for considering a point a minimum
        language (str): Language identifier ('zh' for Chinese, others for NLTK)
        batch_size (int): Batch size for processing
        max_txt_size (int): Maximum text size for processing
        dynamic_merge (str): Dynamic merging strategy ('no' for no merging, others for merging)
        target_size (int): Target size for merging paragraphs
        
    Returns:
        list: List of final chunked paragraphs
    """
    start_time = time.time() 
    if language=='en':
        txt_length=len(sub_text.split())
    else:
        txt_length=len(sub_text)
    if txt_length<=4096:
        new_final_chunks=extract_by_html2text_db_nolist(sub_text,model,tokenizer,threshold,language)
    else:
        new_final_chunks=extract_by_html2text_db_bench(sub_text,model,tokenizer,threshold,language,batch_size,max_txt_size)
    
    if dynamic_merge!='no':
        merged_paragraphs = []  
        current_paragraph = new_final_chunks[0] 
        for paragraph in new_final_chunks[1:]:  
            # Check if adding a new paragraph to the current paragraph exceeds the target size
            if len(current_paragraph) + len(paragraph) <= target_size:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  
                current_paragraph = paragraph  
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph)  

    else:
        merged_paragraphs = new_final_chunks
        
    end_time = time.time()  
    execution_time = end_time - start_time  
    print(f"The program execution time is: {execution_time} seconds.")
    
    return merged_paragraphs