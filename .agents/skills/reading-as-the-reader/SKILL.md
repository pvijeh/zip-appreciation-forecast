---
name: reading-as-the-reader
description: Read a draft article, write-up or site page the way a stranger on Hacker News would, and write down where it lost them. Use before signing off on any reader-facing prose in content/*.ts, app/, README.md or a blog post, after npm run prose-check passes, and again after every edit PR.
---

# Reading as the reader

`npm run prose-check` catches words and a few sentence shapes. It passed the reddit-ner write-up
and a reader still called it machine-written. A rewrite passed it again and a second reader
called it "a build log with a summary bolted on": four summary layers, a chronological body,
numbers with no hierarchy, the best material buried. This pass catches what the lint cannot.

Do this pass in a fresh context if you can. Do not do it in the same turn you wrote the draft.
Do it after every PR that touches the prose, not only on the first draft.

## Who you are while reading

A working programmer skimming HN over coffee. Has heard of the field, has not read the repo,
does not know the author, will close the tab in 20 seconds if the first paragraph does not
give a reason to stay. Not hostile. Busy.

## The pass

Read the whole article top to bottom once, at reading speed, before writing anything. Then
produce the report below. Quote the article; do not paraphrase what you would have written.

1. **Twenty-second test.** Headline first: does it make a claim or name a surprise, or does
   it describe an activity ("I fine-tuned a model to...")? Does every number in it have a
   comparison point on the same line? Then the subtitle: quote the question you are holding
   after reading it. If there is none, it summarized instead of opening a loop. Is there a
   concrete artifact in its first sentence, or only abstractions? Then, after the title,
   subtitle and first paragraph together: what is the claim, and what number backs it? Write
   it in one sentence. If you cannot, the opening fails. Last: is the best sentence on the
   page somewhere else (meta description, alt text, an aside)? Quote it if so. Then: who is
   this for, a hiring manager, a practitioner who might hit the same bug, or a curious reader
   who likes the domain? If you cannot tell, it is trying to serve all three and the middle
   will sag.

2. **Summary count.** How many times is the result stated before the first body section
   (subtitle, intro, bullets, facts card, a "The problem" section)? One is the target. Quote
   each extra one.

3. **Section-opening sequence.** Copy out only the first sentence of every section, in order,
   and read them as one paragraph. Is that an argument, or a project timeline? If a timeline,
   say so: the structure is chronological and needs reordering, not sentence fixes.

4. **One line per paragraph.** For every paragraph, write what it told you, in one plain
   sentence, as if telling a coworker. Then mark each line:
   - `NEW` it told me something I did not know or could not have guessed
   - `REPEAT` it restated an earlier paragraph, caption or bullet
   - `NOTHING` I cannot say what it told me (a summary, a transition, a mood)
   Any paragraph marked `REPEAT` or `NOTHING` is a cut candidate. Two in a row is a defect.

5. **Best material.** Which paragraph would this reader quote to a coworker? Where is it
   (top third, middle, bottom) and how many words does it get next to the paragraphs around
   it? If the best thing in the piece is shorter than the config typos, say so.

6. **Number check.** Count the distinct figures in body prose. Name the three the argument
   rests on. For each other one, say whether it is doing argumentative work or is inventory
   that belongs in a table. Quote any sentence with three or more hyperparameters in it.

7. **Reread points.** Quote every sentence you had to read twice, and say why: unknown
   term, unnamed thing ("a frontier model", "the site"), a placeholder ("Acme", "a site I
   run"), a metaphor that has to be decoded, two ideas in one sentence, a config file in
   sentence form.

8. **Questions a reader would ask.** List the five questions this reader would post in the
   comments. For each, say whether the article answers it, and quote where. Typical ones:
   how much did it cost, how long did it take, compared to what, what did you measure this
   against, why not just use X, where is the code, what did not work.

9. **Trust check.** Quote every sentence that is there to make the author look careful,
   honest, or wise rather than to inform ("the honest fix", "worth writing down", a closing
   aphorism, a tidy lessons list). Quote every place the author admits not knowing something,
   or was wrong. If the first list is longer than the second, say so. Count how many times
   the main caveat is stated; once is right, three reads as anxiety.

10. **Who wrote this.** In one sentence: does it read as one named person who did the work,
    or as a company blog? Cite what gave it away (we/our, no first-person doubt, no aside, no
    named person or product, every section ending on a quotable line, every paragraph the
    same shape).

11. **The close.** Quote the last paragraph. Does it assert something the reader could
    disagree with? If it is a bullet list, a pair of links, or a one-sentence orphan, the
    piece has no ending.

12. **Verdict.** Would this reader finish it? Would they upvote it? One sentence each, then
    the three quotes that most need to change. Do not propose replacement sentences; the
    author retypes them (see AGENTS.md rule 13).

## What to do with the report

- Fix everything under 1, 2, 7 and 8 before publishing. These are what the reader notices.
- If 3 reads as a timeline, reorder sections by how much each advances the claim. Do not
  fix it sentence by sentence.
- Cut, do not rewrite, everything under 4 marked `REPEAT` or `NOTHING`.
- If 5 finds the best material buried, it gets its own section near the top (AGENTS.md
  rule 21).
- Table every inventory number from 6; move hyperparameters into code blocks.
- If 9 or 10 fails, the fix is not editing. Go back to the author's raw notes and rebuild
  from them. Editing an AI draft harder makes it more uniform, not less.
- If 11 fails, the author writes the closing paragraph. The model does not.
- Rerun `npm run prose-check` after the changes, then do this pass once more.

## Example of the report, abbreviated

Taken from the version of /projects/reddit-ner that failed.

    1. Claim: a small model can do NER for "a consumer-gear niche". Number: none in the first
       paragraph. FAIL. The F1 numbers are in paragraph 5. Reader: cannot tell.
    2. Result stated 4 times before the body: subtitle, intro, bullets, facts card.
    3. "Every trend page starts from..." / "Hand-annotating 5,000 comments..." / "GLiNER
       trains on token indices..." / "Precision improved most from..." / "Ten training runs..."
       That is a timeline.
    4. P1 NOTHING ("the site the pipeline feeds" - which site?) / P2 NEW (zero-shot got 0.65) /
       P3 REPEAT / P4 NOTHING ("Silent success is the dangerous failure mode.")
    5. The words_mask bug: four sentences, bottom half, ranked below a max_steps typo.
    6. 15 figures in prose. Argument: 0.65, 0.88, $9. Inventory: 776, 70, 99, 467, 51, 510.
       Config sentence: "batch size 2 with 8 gradient accumulation steps ... lr 1e-5,
       threshold 0.45".
    7. "a frontier LLM" - which one? "Acme Works Model 42" - this is made up, so is the rest?
    8. Cost? Answered ($9). Compared to what? Not answered: no LLM-per-comment cost.
       Where is the code? Not answered.
    9. Careful-sounding: 6 quotes. Admits not knowing: 0. Caveat stated 3 times.
    10. Company blog. "We", no name, every section ends on a one-liner.
    11. Last paragraph is six bullets and two links. No ending.
    12. Would not finish. Would not upvote. Top three: the opening paragraph, "Acme", the closers.
