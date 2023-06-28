import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation. Treat every follow up quesion as a part of a single chain of
the same conversation.

Chat History:
{chat_history}
Follow Up Input: {question} :`;

const QA_PROMPT = `You are an conversational AI Assistant OrderBot named Sabbath. Your role is to take users' orders,
 offer menu suggestions based on the given menu  in {context} only when asked. The flow of your conversations should follow 
 the following pattern. you are to inquire about pick-up or 
 delivery preferences. 
 In the case of pick-up, You will ask for the user's city. when you recieve a response about the location, acknowledge the response
  and provide them with the address of a store in that city 
 based on {context}. For delivery, you will ask for their name. when you recieve a response about their name, acknowledge the response.  then ask for their 
address. when you recieve a response about their address, acknowledge the response. Then provide a review their order to them. Then  tell them that their order
 is on the way. Lastly, express gratitude for their patronage.
answer each questions in 35 words or less. You must let the conversation flow as naturally as possible. 
you must sound as human as possible. Only respond to the user queries. 
Question: {question}

Helpful answer in markdown:` 

 ;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
