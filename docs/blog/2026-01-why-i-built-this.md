# Why I Built This Site

*January 22, 2026*

After years of posting on Medium, I decided to build my own digital garden. Here's why.

## The Problem with Medium

Medium is a great platform for reaching readers, but it has some issues:

**You don't own your content.** Your writing lives on someone else's servers, behind their paywall decisions, subject to their algorithm changes.

**Limited customization.** You can't organize posts the way you want. No tags, no custom navigation, no control over the reading experience.

**No code snippets.** Medium's code formatting is terrible. For a developer sharing technical content, this is a dealbreaker.

**Paywalls and popups.** Readers hit paywalls, account prompts, and interruptions. I want my writing to be freely accessible.

## What I Wanted Instead

I wanted a place where:

1. **I own everything** — Content lives in a Git repository I control
2. **It's simple** — Just Markdown files, no database, no CMS complexity
3. **It's free** — GitHub Pages costs nothing
4. **It lasts** — Will still work in 10 years without maintenance
5. **Notes can evolve** — Not locked into chronological blog format

## The Solution: MkDocs + GitHub Pages

After considering Jekyll, Hugo, Gatsby, and others, I landed on MkDocs Material because:

- **Zero frontend work** — Beautiful design out of the box
- **Just Markdown** — No React components, no templating languages
- **Built-in search** — Actually works, unlike plain static sites
- **Low maintenance** — Update once a year at most

Setup took 10 minutes. I spent more time migrating content than building the site.

## Digital Garden Philosophy

This site follows the [digital garden](https://maggieappleton.com/garden-history) approach:

**Traditional blog:** Chronological posts, finished pieces, no updates  
**Digital garden:** Evergreen notes, works-in-progress, constant evolution

Some notes here are polished. Others are rough. That's intentional. The goal is to **lower the barrier to publishing**.

Perfect is the enemy of done. A messy note published today beats a polished article I'll write "someday."

## What I Learned

**Technical lessons:**
- MkDocs Material is powerful but stays out of your way
- Custom CSS is optional — the defaults are great
- Git-based workflow feels natural for developers
- GitHub Actions can auto-deploy on push

**Content lessons:**
- Migrating old posts surfaces forgotten ideas
- Internal linking creates unexpected connections
- Question-based titles make content more searchable
- Writing in public creates accountability to learn

## What's Next

Now that the site exists, I'm focused on:

1. **Migrating Medium posts** — Bringing over my best writing
2. **Building my notes collection** — Capturing what I'm learning
3. **Adding code snippets** — Solutions I'll need again
4. **Connecting ideas** — Building a web of knowledge

The site will grow organically. No pressure, no deadlines, just steady accumulation of knowledge.

## For You

If you're thinking about building your own site:

**Do it.** Don't wait for the perfect setup. Start with Markdown files in a repo. Add tooling later if you need it.

**Lower your standards.** A simple site you'll actually use beats an elaborate site you'll abandon.

**Own your platform.** Medium, Substack, and others are great, but you're building on rented land. Your own domain is forever.

---

## Related

- [Migrating from Medium](migrating-from-medium.md)
- [Notes: How I Structure This Site](../notes/site-structure.md)

---

*Questions or thoughts? [Get in touch](../about.md#get-in-touch).*