# Extracting Statistical Elements of Experimental Design with Large Language Models

**Dr. Emi Tanaka**
Biological Data Science Institute
Australian National University

**3rd September 2024**

## Title Slide

# Extracting Statistical Elements of Experimental Design with _Large Language Models_

Dr. Emi Tanaka
Biological Data Science Institute
Australian National University

3rd September 2024

## Rise of Large Language Models

* Release of ChatGPT on 30th November 2022 caused widespread use of LLMs

### ChatGPT Heralds an Intellectual Revolution

**25 February 2023**
By Henry Kissinger, Eric Schmidt, and Daniel Huttenlocher

*Generative artificial intelligence presents a philosophical and practical challenge on a scale not experienced since the start of the Enlightenment.*

A new technology bids to transform the human cognitive process as it has not been shaken up since the invention of printing. The technology that printed the Gutenberg Bible in 1455 made abstract human thought communicable generally and rapidly. But new technology today reverses that process. Whereas the printing press caused a profusion of modern human thought, the new technology achieves its distillation and elaboration. In the process, it creates a gap between human knowledge and human understanding. If we are to navigate this transformation successfully, new concepts of human thought and interaction with machines will need to be developed. This is the essential challenge of the Age of Artificial Intelligence.

The new technology is known as generative artificial intelligence; GPT stands for Generative Pre-Trained Transformer. ChatGPT, developed at the OpenAI research laboratory, is now able to converse with humans. As its capacities become broader, they will redefine human knowledge, accelerate changes in the fabric of our reality, and reorganize politics and society.

Generative artificial intelligence presents a philosophical and practical challenge on a scale not experienced since the beginning of the Enlightenment. The printing press enabled scholars to replicate each other's findings quickly and share them. An unprecedented consolidation and spread of information generated the scientific method. What had been impenetrable became the starting point of accelerating query. The medieval interpretation of the world based on religious faith was progressively undermined. The depths of the universe could be explored until new limits of human understanding were reached.

Generative AI will similarly open revolutionary avenues for human reason and new horizons for consolidated knowledge. But there are categorical differences. Enlightenment knowledge was achieved progressively, step by step, with each step testable and teachable. AI-enabled systems start at the other end. They can store and distill a huge amount of existing information, in ChatGPT's case much of the textual material on the internet and a large number of books—billions of items. Holding that volume of information and distilling it is beyond human capacity.

Sophisticated AI methods produce results without explaining why or how their process works. The GPT computer is prompted by a query from a human. The learning machine answers in literate text within seconds. It is able to do so because it has pregenerated representations of the vast data on which it was trained. Because the process by which it created those representations was developed by machine learning that reflects patterns and connections across vast amounts of text, the precise sources and reasons for any one representation's particular features remain unknown. By what process the learning machine stores its knowledge, distills it and retrieves it remains similarly unknown. Whether that process will ever be discovered, the mystery associated with machine learning will challenge human cognition for the indefinite future.

AI's capacities are not static but expand exponentially as the technology advances. Recently, the complexity of AI models has been doubling every few months. Therefore generative AI systems have capabilities that remain undisclosed even to their inventors. With each new AI system, they are building new capacities without understanding their origin or destination. As a result, our future now holds an entirely novel element of mystery, risk and surprise.

Enlightenment science accumulated certainties; the new AI generates cumulative ambiguities. Enlightenment science evolved by making mysteries explicable, delineating the boundaries of human knowledge and understanding as they moved. The two faculties moved in tandem: Hypothesis was understanding ready to become knowledge; induction was knowledge turning into understanding. In the Age of AI, riddles are solved by processes that remain unknown. This disorienting paradox makes mysteries unmysterious but also unexplainable. Inherently, highly complex AI furthers human knowledge but not human understanding—a phenomenon contrary to almost all of post-Enlightenment modernity. Yet at the same time AI, when coupled with human reason, stands to be a more powerful means of discovery than human reason alone.

The essential difference between the Age of Enlightenment and the Age of AI is thus not technological but cognitive. After the Enlightenment, philosophy accompanied science. Bewildering new data and often counterintuitive conclusions, doubts and insecurities were allayed by comprehensive explanations of the human experience. Generative AI is similarly poised to generate a new form of human consciousness. As yet, however, the opportunity exists in colors for which we have no spectrum and in directions for which we have no compass. No political or philosophical leadership has formed to explain and guide this novel relationship between man and machine, leaving society relatively unmoored.

ChatGPT is an example of what is known as a large language model, which can be used to generate human-like text. GPT is a type of model that can be automatically learned from large amounts of text without the need for human supervision. ChatGPT's developers have fed it a massive amount of the textual content of the digital world. Computing power allows the model to capture patterns and connections.

The ability of large language models to generate humanlike text was an almost accidental discovery. These models are trained to be able to predict the next word in a sentence, which is useful in tasks such as autocompletion for sending text messages or searching the web. But it turns out that the models also have the unexpected ability to create highly articulate paragraphs, articles and in time perhaps books.

ChatGPT is further specialized beyond a basic large language model, using feedback from humans to tune the model so that it generates more natural-seeming conversational text, as well as to try to contain its propensity for inappropriate responses (a substantial challenge for large language models). ChatGPT instantaneously converts its representations into unique responses. The ultimate impression on a human conversant is that the AI is relating stationary collections of facts into dynamic concepts.

ChatGPT's answers, statements and observations appear without an explanation of where they came from and without an identifiable author. On its face, ChatGPT has no discernible motive or bias. Its outputs are complex, but its work is astonishingly rapid: In a matter of seconds, it can produce answers that coherently explain a high-level topic. They are not simply copied from the text in the computer's memory. They are generated anew by a process that humans are unable to replicate. It is able to incorporate hypotheticals and nonobvious psychological inferences. It can prioritize among billions of data points to select the single set of 200 words that is most relevant (or will appear most relevant to a human).

## Experimental Design Extraction Task

The presentation focuses on extracting statistical elements from experimental design descriptions using LLMs. The task involves:

### Input Format
Experimental design descriptions from research papers, such as:

**Example 1:**
> "A three–year in–situ field experiment was conducted in Zhongjiang County, Deyang City, Sichuan Province (31.03 °N, 104.68 °E) from 2017 to 2019 maize growing seasons. Before sowing, soil samples were collected from the 0–30cm soil layer using the diagonal soil sampling method. Soil samples were air–dried and sieved to remove undecomposed plant material. The samples obtained were used for the analysis of soil organic matter, total N, Alkaline N, Available P, Available K, pH and the results were shown in Table 1. It was a fallow land in the winter of 2017–2018, and wheat was firstly planted before sowing maize in 2019, resulting in lower soil N content. The experimental materials were the pre–screened low N–tolerant maize variety ZhengHong311 (ZH311) and low N–sensitive maize variety XianYu508 (XY508) (Li et al., 2020, Wu et al., 2022). Seeds were provided by Sichuan Agricultural University Zhenghong Seed Co. and Tieling Pioneer Seed Research Co. respectively."

**Example 2:**
> "This study examined the effects of biochar application on soil health and crop yields over 5 years at a research site near Sidney, NE. The experiment involved a split-plot design with four replications, two main factors (biochar and N fertilizer), and three levels of N fertilizer application (0, 84, and 168 kg N ha^-1). Five levels of biochar applied once at 0, 3.125, 6.25, 12.5, and 25 Mg ha−1, and the crop rotation included proso millet, yellow field peas, winter wheat, and sunflower. The study found that N fertilizer was only applied in three out of five years due to the legume crops being planted. Biochar was incorporated into the soil at a depth of 15 cm and had characteristics such as high carbon content (560 g kg^-1) and total ash (390 g kg^-1)."

### Output Format
Structured JSON extraction of statistical elements:

**Example 1 Output:**
```json
{
  "response": ["total N", "Alkaline N", "Available P", "Available K", "pH"],
  "treatments": {
    "maize varieties": [
      "low N-tolerant variety ZhengHong311 (ZH311)",
      "low N-sensitive variety XianYu508 (XY508)"
    ]
  },
  "units": "soil samples"
}
```

**Example 2 Output:**
```json
{
  "aim": "examine the effects of biochar application on soil health and crop yields",
  "response": "yield",
  "design_name": "split plot design",
  "treatments": {
    "application": ["biochar", "N fertilizer"],
    "N_fertilizer": ["0 kg N ha^-1", "84 kg N ha^-1", "168 kg N ha^-1"],
    "biochar": ["0 Mg ha−1", "3.125 Mg ha−1", "6.25 Mg ha−1", "12.5 Mg ha−1", "25 Mg ha−1"]
  },
  "units": "60 plots over 5 years"
}
```

## Additional Examples

### Example 3: Randomized Block Design
**Input:**
> "The study was conducted at Dajin Town in Hubei Province, China, featuring warm temperatures and a sub-humid continental monsoon climate. The sandy loam soil had a pH of 5.15, 29.2 g/kg organic matter, 1.46 g/kg total nitrogen, 6.37 mg/kg available phosphorus, and 38.46 mg/kg available potassium. The experiment involved four treatments: conventional tillage without straw incorporation (CT), conventional tillage with straw incorporation (CTS), no-tillage without straw mulching (NT), and no-tillage with straw mulching (NTS), each replicated three times. The plots were assigned randomly within blocks, considering potential covariates like soil type and topography."

**Output:**
```json
{
  "design_name": "randomised block design",
  "treatments": {
    "tillage": [
      "conventional tillage without straw incorporation (CT)",
      "conventional tillage with straw incorporation (CTS)",
      "no-tillage without straw mulching (NT)",
      "no-tillage with straw mulching (NTS)"
    ]
  },
  "units": "plots within blocks"
}
```

### Example 4: Simple Treatment Comparison
**Input:**
> "The field experiment was conducted at the Longkang Farm in Anhui Province, China, with a wheat and maize rotation cropping system. The experiment involved three tillage treatments: no-tillage (NT), rotary tillage (RT), and deep ploughing (DP). The tillage treatments were applied before the wheat season each year. The experiment was conducted over six crop seasons from 2017 to 2022, with detailed records of rainfall, planting, harvest times, and growth periods for both wheat and maize crops. The soil at the experimental site was a Vertisol with a particle density of 2.69 g cm−3 and a particle size distribution of 80% sand, 541% silt, and 379% clay."

**Output:**
```json
{
  "treatments": {
    "tillage": ["no-tillage (NT)", "rotary tillage (RT)", "deep ploughing (DP)"]
  },
  "units": "seasons from 2017 to 2022"
}
```

### Example 5: Split-Split Plot Design
**Input:**
> "A split-split plot designed field experiment with a rice-wheat cropping system was conducted from May 2020 to February 2022. The study involved various treatments for both rice and wheat crops. For rice, two varieties (PR122-V1 and PR126-V2) were tested in the main plots, while irrigation levels (I1 and I2) and nitrogen levels (120-N1 and 90-N2 kg N ha−1) were evaluated in subplots and subplot-subplots, respectively. Similarly, for wheat, two varieties (Unnat PBW343-V1 and Unnat PBW550-V2) were tested in the main plots, with irrigation levels (I1 and I2) and nitrogen levels (120-N1 and 90-N2 kg N ha−1) being evaluated in subplots and subplot-subplots. The experiment was replicated three times for each treatment, with a plot size of 12 × 6 m. The results showed differences in yields between the rice and wheat varieties, with PR126-V2 yielding higher than PR122-V1, and Unnat PBW550-V2 yielding lower than Unnat PBW343-V1 under non-water stress conditions."

**Output:**
```json
{
  "design_name": "split-split plot design",
  "treatments": {
    "rice": {
      "varieties": ["PR122-V1", "PR126-V2"],
      "irrigation_levels": ["I1", "I2"],
      "nitrogen_levels": ["120-N1", "90-N2 kg N ha−1"]
    },
    "wheat": {
      "varieties": ["Unnat PBW343-V1", "Unnat PBW550-V2"],
      "irrigation_levels": ["I1", "I2"],
      "nitrogen_levels": ["120-N1", "90-N2 kg N ha−1"]
    }
  },
  "units": "plots"
}
```

## Model Performance Results

### Result 3: llama3:70b with Prompt #1 is best

(out of those compared)

| Name              | Prompt | π̂_Pass+ | π̂_Semi-correct+ | π̂_Correct |
|-------------------|--------|---------|-----------------|-----------|
| llama3:70b (#1)   | #1*    | 0.872   | 0.720           | 0.710     |
| llama3.1:70b (#1) | #1*    | 0.769   | 0.639           | 0.612     |
| llama3:8b (#1)    | #1*    | 0.824   | 0.578           | 0.559     |
| llama3.1:8b (#1)  | #1*    | 0.769   | 0.578           | 0.559     |
| llama3:70b (#2)   | #2     | 0.584   | 0.500           | 0.464     |
| llama3.1:70b (#2) | #2     | 0.502   | 0.426           | 0.414     |
| llama3.1:8b (#2)  | #2     | 0.502   | 0.365           | 0.306     |
| llama3:8b (#2)    | #2     | 0.394   | 0.261           | 0.217     |

*Has 3 replicates

## Future Directions

* LLM can potentially:
  * help us analyse large scale of literature, in this case, dissecting common experimental structures and usage of experimental designs in practice
  * build a experimental design "recommender" or "second opinion" by learning the domain context
  * and more
* an R package that uses LLM to extract statistical elements of experimental designs
  * _to come_
* Refining the model and evaluation on validation data required...

**Did you want to collaborate/coauthor?**
Come chat or reach out at emi.tanaka@anu.edu.au!

## Postdoc Opportunities

* **5+ postdoc** opportunities coming up at **ANU Biological Data Science Institute**:
  * _AAGI - Analytics for the Australian **Grains** Industry research collaboration_
    with Emi Tanaka, Eric Stone, Alan Welsh and Francis Hui
  * _ARC Training Centre in **Plant Biosecurity**_
    with Eric Stone and others
  * _MACSYS - ARC Centre of Excellence for the Mathematical Analysis of **Cellular Systems**_
    with Eric Stone and others

Come chat or reach out at emi.tanaka@anu.edu.au

---

* This slide is available at emitanaka.org/slides/AASC2024

**Source:** [emitanaka.org/slides/AASC2024](https://emitanaka.org/slides/AASC2024/#/title-slide)
