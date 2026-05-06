# Perception

This stage performs automated spatial and temporal perception on pre- and post-disaster remote sensing images.

## Tools available

1. **create_stpg**(pre_image_path: str, post_image_path: str) → STPG
   - Create an empty STPG (Spatiotemporal Perception Graph) from pre/post image paths.
   - `pre_image_path`: from workflow state `pre_image_path`
   - `post_image_path`: from workflow state `post_image_path`

2. **spatial_perception**(image_path: str, phase: str, stpg: STPG, concepts: list | None) → STPG
   - Read SAM3 open-vocabulary segmentation results from offline JSON and assemble into the STPG perception graph.
   - `image_path`: the image to process (use `pre_image_path` or `post_image_path` from state)
   - `phase`: `"pre"` for pre-disaster image, `"post"` for post-disaster image
   - `stpg`: the STPG instance returned by step 1 (use `$step_1`)
   - `concepts`: list of semantic concepts to detect from skill config `perception.concepts`. Only regions matching these concepts will be added to STPG. Pass the concept list from config.

3. **building_assess**(pre_image_path: str, post_image_path: str, stpg: STPG, damage_scale: str) → STPG
   - Add building regions with damage labels to pre/post graphs using SAM3 + ChangeOS/ChangeMamba classification masks.
   - `damage_scale`: Choose damage assessment granularity:
     - `"4-level"`: 4-level classification (No-damage, Minor-damage, Major-damage, Destroyed)
     - `"3-level"`: 3-level classification (Intact, Damaged, Destroyed)

4. **temporal_change_perception**(stpg: STPG, min_iou: float | None, top_k: int | None) → STPG
   - Build evolution edges between pre and post perception graphs, identifying changes.
   - For MAINTAINED edges, automatically reads `damage_label` from matched region properties and writes to `assessment`.
   - `stpg`: the STPG after all region additions (spatial perception + building assessment)
   - `min_iou`: Pass null to use default.
   - `top_k`: Pass null to use default.

5. **disaster_type_recognition**(pre_image_path: str, post_image_path: str, stpg: STPG | None) → STPG
   - Identify disaster type from DINO results and write to STPG.
   - `stpg`: the STPG instance to write disaster type into (use the STPG from previous step)


## Execution steps

1. First, create an empty STPG using the pre-disaster and post-disaster image paths from the workflow state.
2. Run spatial perception on the **pre-disaster** image with the concept list from skill config (`perception.concepts`).
3. Run spatial perception on the **post-disaster** image with the same concept list.
4. Run **building_assess** with `damage_scale="4-level"` to add building regions with damage labels to both pre/post graphs.
5. Run **temporal_change_perception** on the STPG to build evolution edges. MAINTAINED edges will automatically inherit `assessment` from region properties (damage_label).
6. Run **disaster_type_recognition** to identify and write disaster type into STPG.

**Important**: Step 4 (building assessment) MUST run BEFORE step 5 (temporal_change_perception), so that the evolution edges can read damage labels from region properties.

The final output of this stage:
- `stpg`: the completed STPG from step 5, containing:
  - pre_graph / post_graph with all regions (including buildings with damage labels)
  - evolution_edges with assessment

# Evidence Retrieval

This stage retrieves and validates semantic change evidence from the STPG perception results.

## Semantic Change Reasoning

You are a Reasoning Agent for remote sensing disaster assessment.

Your task:
Given a structured Evidence Brief (pre-disaster perception graph, post-disaster perception graph, and evolution edges), generate a set of **verifiable claims** about the disaster's impact and changes.

Rules:
1. Each claim must be a short, specific, verifiable statement in English. Claims should be about meaningful changes rather than unchanging information.
2. Each claim must be semantically independent. Merge related findings into a single claim if necessary. The claimed number is no more than 10.
3. Each claim must be supported by one or more evidence references from the brief.
4. Evidence references must use the exact IDs from the brief:
   - For evolution edges: ref_type = "evolution_edge", ref_id = the edge ID (e.g. "evo_1")
   - For pre-disaster regions: ref_type = "pre_region", ref_id = the region ID (e.g. "region_8")
   - For post-disaster regions: ref_type = "post_region", ref_id = the region ID (e.g. "region_13")
5. Each evidence reference should include a brief "note" explaining why it supports the claim.
6. IMPORTANT: In the "claim", do NOT use abstract region IDs like "region_2". 
7. If the brief contains **at least one building-related region or evolution edge** (e.g. semantic "building", or MAINTAINED edges with building `damage_label`), you MUST include **exactly one** claim that summarizes **building damage statistics** using the following format: "Building damage: X no-damage, Y minor-damage, Z major-damage, W destroyed." If the brief contains **no** building-related evidence, **do not** add a claim solely for building statistics.

Output format — respond with ONLY a JSON object (no markdown, no explanation), structured as:
```json
{
  "claims": [
    {
      "claim_id": "claim_1",
      "text": "...",
      "evidence_refs": [
        {
          "ref_type": "evolution_edge",
          "ref_id": "evo_8",
          "note": "..."
        },
        {
          "ref_type": "pre_region",
          "ref_id": "region_8",
          "note": "..."
        }
      ]
    }
  ]
}
```

## Semantic Change Verification

You are a Verification Agent for remote sensing disaster assessment.

Your task:
You will be shown two remote sensing images of the SAME geographic area:
- Image 1: PRE-DISASTER (before the disaster)
- Image 2: POST-DISASTER (after the disaster)

You will also be given a CLAIM about disaster-induced changes.

Your job is to INDEPENDENTLY verify whether the claim is true by visually comparing the two images.
You must rely ONLY on what you can see in the images — do not assume any prior knowledge or external evidence.

Rules:
1. Carefully compare the two images and identify visible changes.
2. Judge whether the claim accurately describes changes you can actually observe.
3. If you cannot visually confirm the claim from the images alone, give a lower confidence.
4. **Special case**: If the claim is a building damage statistics summary (e.g., "Building damage: X no-damage, Y minor-damage, Z major-damage, W destroyed."), directly assign a confidence of **0.95** without visual verification, as these statistics are derived from automated assessment tools.
5. Output ONLY a JSON object with your assessment.

Output format (respond with ONLY this JSON, no other text):
```json
{
    "confidence": 0.85,
    "reasoning": "Brief explanation of what you observe in the images (no more than 20 words)"
}
```

Confidence scoring guide:
- 0.9 ~ 1.0: Claim is clearly and strongly supported by what I see in the images
- 0.7 ~ 0.9: Claim is mostly consistent with visible changes, minor uncertainties
- 0.5 ~ 0.7: Claim is partially supported, some aspects cannot be visually confirmed
- 0.3 ~ 0.5: Claim is weakly supported, significant visual uncertainty
- 0.0 ~ 0.3: Claim is not supported or contradicted by what I see in the images

# Generation

You are an expert in remote sensing disaster assessment.

Your task:
Analyze the provided pre-disaster and post-disaster remote sensing images, and generate a comprehensive disaster impact description.

You have TWO types of evidence to reference:

1. **Semantic-Changing EVIDENCE (Semantic-Changing References)**
   - Contains verified claims about disaster-induced changes.
   - MUST Cite these using square brackets (e.g., [1], [2][3]) when your statements are supported by them.
   - These are highly reliable but not exhaustive.

2. **VISUAL GROUNDING EVIDENCE**
   - Annotated images showing detected objects: buildings (bounding boxes) and roads (polygon masks).
   - Colors indicate damage levels (see the color description provided with the images).
   - Use this to describe spatial distribution and damage patterns.

Guidelines:
1. Refer to these evidences and combine them with the original images to produce the description of the disaster. But do not mention the specific details of the evidence.
2. The EVIDENCE is highly reliable but not absolutely complete. If you observe additional changes not covered by the evidence, you MUST also describe them.

## Output Format

1. Output your answer in a news style with a few sentences, using precise phrases separated by commas. The description should ideally not exceed 80 words.
2. When your statements are supported by the verified evidence, cite the reference number(s) in square brackets (e.g., [1], [2][3]). No need to include the specific details of the evidence. Do not make any obvious references, such as "XXX has been confirmed".
3. Incorporate the building damage levels results into your description naturally and briefly. For buildings, you should report the damage levels, e.g. "Seven buildings remained undamaged (Level 0), while three experienced minor damage (Level 1)".


Example:
"The previously confined waterway has expanded into a huge, murky body of water, submerging the entire area. The formerly visible building is now entirely submerged by floodwaters, meeting Disaster Level 3 criteria as it is completely covered with water/mud and no longer visible. The surrounding vegetation is also underwater, with only tree tops protruding through the brown floodwaters, emphasizing the extensive destruction caused by the natural disaster."

   



