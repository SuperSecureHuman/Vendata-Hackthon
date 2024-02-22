# Results of the models

2 Models - Phi 2 and Code Llama here.

There is some stop token issue with Phi - neverthless, the outputs are good enuf.

Both lora adapters will be put on huggingface hub and link will be put here shortly

I am not very well versed with web tech, so I can't test this model extensively.

## Phi 2

![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/854043f1-38b1-4c54-8ab7-2ec171a5afcc)



```py
inputs = tokenizer('''// Function to calculate the Fibonacci sequence up to n terms
function fibonacci(n) {
  let sequence = [0, 1];

   ''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=100, early_stopping=True)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/f91fe79b-1b23-418a-b731-11d0e019bd10)


```py
inputs = tokenizer('''

/**
 * MySamplePage: A sample React page component.
 */

// Import React and other required modules
import React, { useState, useEffect } from 'react';

// Import any additional components or styles, if needed
import './MySamplePage.css';

const MySamplePage = () => {
  // Define any state or effect hooks, if needed
  const [count, setCount] = useState(0);

  useEffect(() => {
    document.title = `You clicked ${count} times`;
  }, [count]);

  // Define component methods, if needed
  const increment = () => setCount(count + 1);

  // Render the component markup


   ''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=500, early_stopping=True)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```


![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/ee73a556-e7ca-4409-bea8-3f9cad196376)

## Code LLama 

![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/266fb04f-a68e-418e-aa2c-20b0e46be3ac)


![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/8cde820c-5bec-4cc9-8f5f-7222472246c0)

```py
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "./vendata-train"
config = PeftConfig.from_pretrained(peft_model_id)


torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", trust_remote_code=True)



# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

inputs = tokenizer('''

/**
 * MySamplePage: A sample React page component.
 */

// Import React and other required modules
import React, { useState, useEffect } from 'react';

// Import any additional components or styles, if needed
import './MySamplePage.css';

const MySamplePage = () => {
  // Define any state or effect hooks, if needed
  const [count, setCount] = useState(0);

  useEffect(() => {
    document.title = `You clicked ${count} times`;
  }, [count]);

  // Define component methods, if needed
  const increment = () => setCount(count + 1);

  // Render the component markup


   ''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=500, early_stopping=True)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

![image](https://github.com/SuperSecureHuman/Vendata-Hackthon/assets/88489071/25df0cee-c04c-4381-9f56-d281d58aede8)

