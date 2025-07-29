# Dr. Emi Tanaka - AASC 2024 Presentation

## Overview

This directory contains the content from Dr. Emi Tanaka's presentation at the Australian Applied Statistics Conference (AASC) 2024, titled "Extracting Statistical Elements of Experimental Design with Large Language Models."

## Presentation Details

- **Title**: Extracting Statistical Elements of Experimental Design with Large Language Models
- **Author**: Dr. Emi Tanaka
- **Affiliation**: Biological Data Science Institute, Australian National University
- **Date**: 3rd September 2024
- **Conference**: Australian Applied Statistics Conference (AASC) 2024
- **Source**: [emitanaka.org/slides/AASC2024](https://emitanaka.org/slides/AASC2024/#/title-slide)

## Key Contributions

### 1. LLM-Based Experimental Design Extraction
Dr. Tanaka's work demonstrates how large language models can be used to automatically extract structured statistical information from experimental design descriptions in research papers.

### 2. Structured Output Format
The presentation shows how to convert natural language experimental descriptions into structured JSON format containing:
- Response variables
- Treatment factors and levels
- Experimental design type (e.g., "split plot design", "randomised block design")
- Experimental units
- Study aims

### 3. Model Performance Evaluation
Comprehensive evaluation of different LLM models (llama3:70b, llama3.1:70b, etc.) on the experimental design extraction task, with llama3:70b showing the best performance.

## Relevance to stats-agents Project

This work is highly relevant to the stats-agents project for several reasons:

### 1. **Automated Model Building**
- Demonstrates how LLMs can parse experimental descriptions to extract statistical structure
- Provides a foundation for automatically generating appropriate statistical models based on experimental design

### 2. **R2D2 Integration Potential**
- The structured extraction could inform R2D2 prior specification
- Experimental design type could guide R2D2 variant selection (Shrinkage vs GLM vs M2)
- Treatment structure could inform variance component calculation

### 3. **Laboratory Data Applications**
- Many examples focus on agricultural/biological experiments
- Shows how to handle complex experimental designs (split-plot, split-split plot)
- Demonstrates extraction of multiple grouping factors

### 4. **Future Development**
- Dr. Tanaka mentions developing an R package for this purpose
- Potential for collaboration or integration with stats-agents
- Could inform the agent's ability to automatically parse experimental descriptions

## Files

- `slides.md`: Complete content from the presentation slides
- `README.md`: This overview file

## Contact Information

For collaboration opportunities:
- **Email**: emi.tanaka@anu.edu.au
- **Institution**: Australian National University, Biological Data Science Institute

## Related Work

Dr. Tanaka's work complements the R2D2 framework by providing:
1. **Input parsing**: How to extract experimental structure from text
2. **Design recognition**: Automatic identification of experimental design types
3. **Structured output**: Standardized format for experimental information

This could be integrated with the R2D2 model building process to create a complete pipeline from experimental description to fitted probabilistic model.
