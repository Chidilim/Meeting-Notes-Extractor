import pytest 
import json

@pytest.fixture
def meeting():
    with open ("data/QMSum/data/ALL/train/Bmr009.json", "r") as f:
        meeting_dict = json.load(f)
    return meeting_dict["meeting_transcripts"][0]
@pytest.fixture
def meeting_transcript(meeting):  
    meeting_transcript_list = []
    #creating each turn speech and attaching it to the speaker
    
    speaker,content = meeting.values()
    meeting_transcript_list.append(f"{speaker} : {content}")
    return meeting_transcript_list

def test_transcript_construction(meeting,meeting_transcript):
    speaker = meeting["speaker"]
    content = meeting["content"]
    combined = f"{speaker} : {content}" 
    assert combined == meeting_transcript[0]

