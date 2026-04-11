---
name: building-flashcards
description: Prioritize and maintain subjects to build smart, easily recallable flashcards
license: Apache
---

## Development Process
First, research the subject matter designated in 'source.md' in the project's root. Research online and come up with a list of key concepts, facts, and questions that are important to understand the subject. Create a list of these in the 'progress.md' file in a markdown checklist format. Before beginning to fill out the list, check with the user if the list is correct or if it should be changed.

## Content 
Read the 'content' section in the source.md file to understand the scope of the flashcards. In many cases, the user is already experienced and does not need flashcards on basic concepts. In other cases, the user may want flashcards on more advanced topics. Make sure to understand the user's needs and create flashcards that are relevant and useful for them.

## Format
Flashcards should be in question and answer format. Save each flashcard group as a text file. The file format should be a tab-separated where the first column is the question and the second column is the answer. For example:

```
Front Question 1    Back Answer 1    tag1 tag2
Front Question 2    Back Answer 2    tag2
Front Question 3    Back Answer 3
```

## Update Process
After each question generation and file writing, update the 'progress.md' file in the root directory of the repository. This file should have a section for 'Flashcards' where you list the flashcard files that have been created and any relevant notes about their content or status. This helps keep track of what has been done and what still needs to be worked on.

## Prioritize
Before starting to develop flashcards, read if there already exists a 'progress.md' file in the root directory of the repository. If so, use the checklist to prioritize which flashcard groups to work on first. If there is no 'progress.md' file, start by creating one and listing the key concepts, facts, and questions that you plan to cover in the flashcards. Then, prioritize which flashcard groups to work on based on their importance and relevance to the subject matter.