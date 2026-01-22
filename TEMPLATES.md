# Content Templates

Copy these templates when creating new content.

---

## Blog Post Template

**Filename:** `blog/descriptive-title.md`

```markdown
# Your Blog Post Title

*Month Day, Year*

Opening paragraph: Hook the reader with why this topic matters or what problem you're solving.

## Section 1: The Problem/Context

Explain the situation or problem you encountered. Give enough context for readers to understand.

## Section 2: Your Solution/Approach

Walk through your solution, learning, or perspective.

### Subsection if Needed

Break complex ideas into digestible pieces.

## What I Learned

Key takeaways or insights. This is often the most valuable part.

## Conclusion

Wrap up with final thoughts or next steps.

---

## Related

- [Link to related note](../notes/topic.md)
- [Link to related snippet](../snippets/code.md)

---

*Questions? [Get in touch](../about.md#get-in-touch).*
```

---

## Note Template (Explanation)

**Filename:** `notes/category/question-or-concept.md`

```markdown
# Understanding [Concept] OR How [Something] Works

Brief answer or summary in 1-2 sentences.

## The Core Idea

Explain the fundamental concept in simple terms. Use an analogy if helpful.

## Why This Matters

Context for why you're writing this note. What problem does this solve?

## How It Works

The main explanation. Use:
- Clear subsections with ##
- Code examples
- Comparisons to familiar concepts

### Example

```language
# Working code example
```

Explain what the code does and why.

## Common Patterns

When and how is this typically used?

## What I Misunderstood

What did you get wrong initially? What do others typically misunderstand?

## When to Use

✅ Good use cases
✅ Another good use case

## When NOT to Use

❌ Bad use case and why
❌ Another bad use case

## Common Mistakes

1. **Mistake one** — Why it's wrong and how to avoid it
2. **Mistake two** — Explanation

## Related Notes

- [Another concept](../other/concept.md)
- [Related topic](related-topic.md)

## Resources

- [Official Documentation](https://example.com)
- [Tutorial or Article](https://example.com)

---

*Last updated: YYYY-MM-DD*
```

---

## Code Snippet Template

**Filename:** `snippets/language-what-it-does.md`

```markdown
# Language: What This Snippet Does

**Problem:** Describe the specific problem this solves in one sentence.

## The Snippet

```language
# Complete, working code
# Include all imports
# Make it copy-paste ready

def example():
    """Include docstrings"""
    pass

# Show usage example
result = example()
```

## How It Works

Brief explanation of the key parts:

1. **Line X** — What this does
2. **Line Y** — Why this is needed
3. **Line Z** — Important detail

## Customization Examples

### Variant 1

```language
# Show how to modify for different use case
```

### Variant 2

```language
# Another common variation
```

## When to Use

✅ Use case one  
✅ Use case two  
✅ Use case three

## When NOT to Use

❌ Bad use case — Explanation  
❌ Another bad use case — Why not

## Improvements to Consider

Optional enhancements:
- Add logging
- Add error handling
- Add type hints
- Performance optimization

## Alternative Approaches

If this doesn't fit your needs, consider:
- **Approach A** — When to use it
- **Library B** — What it offers

## Related

- [Related snippet](other-snippet.md)
- [Concept explanation](../notes/category/concept.md)

---

*Last updated: YYYY-MM-DD*
```

---

## Learning Note Template (Work in Progress)

**Filename:** `notes/category/learning-topic.md`

```markdown
# Learning: [Topic]

I'm currently learning about [topic]. These are rough notes.

## Current Understanding

What I think I know so far (may be incomplete or wrong).

## Key Concepts

- **Concept 1** — Brief explanation
- **Concept 2** — Brief explanation
- **Concept 3** — Brief explanation

## Code Examples

```language
# Examples I'm working through
```

## My Current Questions

- What is the difference between X and Y?
- Why does Z happen?
- How do I handle edge case A?

## Confusions

Things that don't make sense yet:
- Confusion 1
- Confusion 2

## TODO

- [ ] Finish reading Chapter X
- [ ] Build a small example project
- [ ] Write this up properly

## Resources

- [Link to tutorial](https://example.com)
- [Link to documentation](https://example.com)

---

*Started: YYYY-MM-DD*  
*Status: Work in progress*
```

---

## Migrated Medium Post Template

**Filename:** `blog/original-medium-title.md`

```markdown
# Original Title from Medium

*Originally published: Month Day, Year*

!!! info "Migrated from Medium"
    This post was originally published on [Medium](https://medium.com/@you/post-url).
    Migrated here for preservation and accessibility.

[Your Medium post content here, converted to Markdown]

---

## Updates Since Original Publication

*Added: YYYY-MM-DD*

Any new thoughts, corrections, or updates since you originally wrote this.

---

## Related

- [Link to related content](../notes/topic.md)

---

*Last updated: YYYY-MM-DD*
```

---

## Quick Reference: Markdown You'll Use

### Headings
```markdown
# H1 — Page title (use once)
## H2 — Major sections
### H3 — Subsections
```

### Links
```markdown
[Link text](../other-page.md)
[External link](https://example.com)
```

### Code Blocks
````markdown
```python
def example():
    return "Hello"
```
````

### Lists
```markdown
- Unordered item
- Another item

1. Ordered item
2. Another item
```

### Emphasis
```markdown
*italic*
**bold**
`inline code`
```

### Callout Boxes
```markdown
!!! note
    Important information

!!! warning
    Warning or caution

!!! tip
    Helpful tip
```

### Tables
```markdown
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
```

### Blockquotes
```markdown
> This is a quote
```

---

## File Naming Conventions

**Blog posts:** `descriptive-title.md`
- `why-i-built-this-site.md`
- `learning-rust-journey.md`

**Notes (question-based):** `how-does-x-work.md` or `what-is-x.md`
- `how-does-python-async-work.md`
- `what-is-the-difference-between-let-and-const.md`

**Snippets:** `language-what-it-does.md`
- `python-retry-decorator.md`
- `sql-recursive-queries.md`
- `bash-find-large-files.md`

**Use hyphens, not underscores or spaces**

---

## When to Use Each Template

**Blog post:** Polished writing, longer form, personal reflection, tutorials

**Note:** Explaining a concept, answering a question, learning something

**Snippet:** Working code you'll reuse, problem-solution format

**Learning note:** Raw thoughts, work in progress, not ready to polish

---

Save this file as `TEMPLATES.md` in your repo root for easy reference!