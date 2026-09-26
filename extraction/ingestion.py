from schemas.extraction_schema import MeeetingNote, ChunkStrat, ChunkStratCater, ExtractedItem,ExtractedItems
import json
import tiktoken
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    

def get_meeting_data(data: MeeetingNote, strategy:ChunkStrat):
    #function for arranging and setting up meeting data, then we can split and then use to 
    chunks = []
    meeting = data.meeting
    
    #saving all topic dicts in a list
    topics = [item for item in meeting["topic_list"]]
    meeting_transcript = []
    
    #creating each turn speech and attaching it to the speaker
    for meeting_dict in meeting["meeting_transcripts"]:
        speaker,content = meeting_dict.values()
        meeting_transcript.append(f"{speaker} : {content}")
    
    if strategy.chunk_strategy == ChunkStratCater.FIXED:
        chunks = chunk_by_fixed_size(meeting_transcript, chunk_size =2000)
        
    if strategy.chunk_strategy == ChunkStratCater.TOPIC:
        chunks = chunk_by_topic(meeting_transcript,topics)
        
    if strategy.chunk_strategy == ChunkStratCater.ITEM:
        
        EXTRACTION_PROMPT = """You are extracting structured information from a meeting transcript segment.

                    Identify every action item and decision made in this transcript. 
                    - An action item is a task someone committed to doing.
                    - A decision is a conclusion or choice the group settled on.

                    For each one, extract:
                    - type: "action_item" or "decision"
                    - text: a clear, self-contained restatement of it
                    - owner: the person responsible, if stated, otherwise null

                    Transcript segment:
                    {topic_chunk}
                    """
        
        topic_chunks = chunk_by_topic(meeting_transcript,topics)
        chunks = chunk_item(topic_chunks,EXTRACTION_PROMPT)
    
    #if strategy.chunk_strategy == ChunkStratCater.TOPIC_ITEM :
        
    return chunks 
    
            
def chunk_by_fixed_size(meeting_transcript, chunk_size):
    chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")
    meeting_transcript_str = " ".join(meeting_transcript)
    tokens = encoding.encode(meeting_transcript_str)
    for start in range(0,len(tokens), chunk_size):
        token_chunks = tokens[start:start+chunk_size]
        chunks.append(encoding.decode(token_chunks))
        
    return chunks 
  
    
    
def chunk_by_topic(meeting_transcript,topics, max_tokens=1500):
    chunks = []
    for topic in topics:
        topic_title, relevant_text_span = topic.values()
        topic_chunk = topic_title
                
        for start,end in relevant_text_span:
            topic_chunk = topic_chunk + " " + " ".join(meeting_transcript[int(start):int(end)+1])
        
        chunks.extend(token_limit(topic_chunk,max_token_limit=max_tokens))
        
    return chunks 

def chunk_item(topic_chunks, prompt):
    all_items = []
    num_failed_chunks = 0

    for chunk in topic_chunks:
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(topic_chunk=chunk)
                }
            ],
            response_format=ExtractedItems,
        )

        result = response.choices[0].message.parsed

        if result is None:
            print(f"Extraction failed for chunk: {chunk[:50]}...")
            num_failed_chunks += 1
            continue

        all_items.extend(result.items)

    if len(topic_chunks) > 0:
        print(
            f"Chunk failure rate is "
            f"{num_failed_chunks / len(topic_chunks):.2%}"
        )
    else:
        print("No topic chunks were provided.")

    return all_items
        
def token_limit(chunks,max_token_limit = 1500):
    encoding = tiktoken.get_encoding("cl100k_base")
    all_tokens = encoding.encode(" ".join(chunks))
    split_tokens = []
    
    for start in range(0,len(all_tokens),max_token_limit):
        split_tokens.append(encoding.decode(all_tokens[start:start+max_token_limit]))
    
    return split_tokens
    




if __name__ == "__main__":
    
    strategy1 = ChunkStrat(chunk_strategy=ChunkStratCater.FIXED)
    strategy2 = ChunkStrat(chunk_strategy=ChunkStratCater.TOPIC)
    
    
    with open("/Users/dillyejeh/Documents/Career/August 2026 cycle/Meeting Notes Extractor/data/QMSum/data/ALL/train/Bed004.json", "r") as f:
        meeting = json.load(f)
    
    data = MeeetingNote(meeting=meeting)
    
    meeting_transcript = []
    topics = [item for item in meeting["topic_list"]]
            
        #creating each turn speech and attaching it to the speaker
    for meeting_dict in meeting["meeting_transcripts"]:
        speaker,content = meeting_dict.values()
        meeting_transcript.append(f"{speaker} : {content}")
    
    print("Printing combine meeting trasncript\n")
    print(meeting_transcript[0:5])
    print("\n")
    
    print("Printing chunked by fized size")
    fixed_chunks = chunk_by_fixed_size(meeting_transcript, 256)
    assert fixed_chunks == get_meeting_data(data, strategy1)
    print(fixed_chunks[0:5])

    print("\n")

    print("Printing chunked by topic")
    topic_chunks = chunk_by_topic(meeting_transcript, topics)
    assert topic_chunks == get_meeting_data(data, strategy2)
    print(topic_chunks[0:5])
    print()
        
    
#def chunk_by_topic_extracted_form:
    
    
#SO WE ARE GOING TO READ IN THE FILES AND THEN VERIFY THAT IT'S A JSON
#Load the JSON as a string 
#PASS IT INTO THE INGEST FUNCTTION WHICH IS THE MANGER FOR ALL THE INGESTION FUNCTION: CHUNK DATA, THEN EMBEDDING.. , STORIG THE DATA