# TANQ: An Open Domain Dataset for Table-ANswered Questions
The first open domain question answering dataset where the answers require building tables from information across multiple sources.


## Dataset
Please click the links below to download directly.

- Test set: https://storage.mtls.cloud.google.com/tanq/data/v1/test.jsonl
- Dev set: https://storage.mtls.cloud.google.com/tanq/data/v1/dev.jsonl


## Description
Description of data elements:

```
"init_qid": question identifier,
"init_question": Original question from QAMPARI before adding attributes,
"ext_question": New question with added attributes,
"ext_question_cleaned": Question without attributes pareatheness,
"ext_question_rephrased": Question rephrased by a language model, this is the question field that should be used,
"question_properties": List of the added attributes into question,
"answer_list": List of rows answers
      "init_answer_wikidata_id": Answer ID in wikidata,
      "init_answer_wikipedia_id": Answer ID in wikipedia,
      "init_answer_composed":
      "extension_answer": List of properties (columns)
            "extension_property_id": Wikidata identifier for the property,
            "extension_property_label": Name for the property,
            "extension_entity": Dict of meta data of current extension answer,
            "extension_wikidata_id": Wikidata ID for the value of the property,
            "extension_wikipedia_id": Wikipedia ID for the value of the property,
            "proof": List of evidence for the property in Wikipedia documents, detailed information could be found below,
      "init_answer_proof": List of initial proofs from QAMPARI
            "proof_text": Content of proof,
            "found_in_url": URL link where the proof is found,
      "filter_pass": Flag of whether this answer pass filters in question or not,
      "instance_type": Type of this answer, extracted from 'instance of' section from corresponding wikidata page,
"extended_types": List of successed extension types for question,
"answer_table": Golden answer table for the extended question.
```

There are three types of proofs:

- InfoboxProof

      ```
      "key": Name of the attribute this proof is related to,
      "value": Answer of the attribute in 'key',
      "section": Section name where the proof is found,
      "parent_section": Parent section's name of 'section',
      "found_in_url": URL link where the proof is found,
      "index": Index of the inforbox in original page,
      "proof_type": Fixed string: 'infobox',
      "eval_result": Evaluation result got from LLM of whether this proof is verified or not.
      ```

- TableProof

      ```
      "rows": List of row data
            "cells": List of cell data
                  "cell_value": Content of cell,
      "caption": Caption of the table if exists,
      "section": Section name where the proof is found,
      "parent_section": Parent section's name of 'section',
      "found_in_url": URL link where the proof is found,
      "index": Index of the table in original page,
      "proof_type": Fixed string: 'table'
      "eval_result": Evaluation result got from LLM of whether this proof is verified or not.
      ```

- TextProof

      ```
      "text": Context of proof,
      "section": Section name where the proof is found,
      "parent_section": Parent section's name of 'section',
      "found_in_url": URL link where the proof is found,
      "index": Index of the section content in original page,
      "proof_type": Fixed string: 'text',
      "eval_result": Evaluation result got from LLM of whether this proof is verified or not.
      ```

## Citing this work
```latex
@article{TANQ-2024,
      title={TANQ: An open domain dataset of table answered questions},
      author={Mubashara Akhtar and Chenxi Pang and Andreea Marzoca and Yasemin Altun and Julian Martin Eisenschlos},
      year={2024},
}
```


## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
