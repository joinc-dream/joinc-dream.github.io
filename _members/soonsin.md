---
title: "Soonsin's Portfolio"
layout: "single"
author_profile: true
permalink: /members/soonsin/
author: soonsin
---

<style>
  /* Hide the social media link buttons */
  .author__urls {
    display: none;
  }
  /* Adjust avatar size */
  .author__avatar img {
    max-width: 120px;
  }
  /* Adjust font sizes */
  .author__name {
    font-size: 1.2em;
  }
  .author__bio {
    font-size: 0.9em;
  }
  /* Align sidebar content with main content */
  .sidebar {
    margin-top: 4.5em; /* Adjust this value as needed */
  }
</style>

{%- assign author_id = page.author -%}
{%- assign portfolio = site.data.portfolios[author_id] -%}

### {{ portfolio.about.title }}

*{{ portfolio.about.description }}*

<ul>
{% for item in portfolio.about.details %}
  <li><strong>{{ item.key }}:</strong> {{ item.value }}</li>
{% endfor %}
</ul>

## Skills

<ul>
{% for skill in portfolio.skills %}
  <li>{{ skill.name }} ({{ skill.level }}%)</li>
{% endfor %}
</ul>

---

## Resume
### Summary

**{{ portfolio.resume.summary.name }}**

*{{ portfolio.resume.summary.description }}*

### Education

{% for item in portfolio.resume.education %}
**{{ item.degree }}**
*{{ item.school }} ({{ item.period }})*
{% endfor %}

### Professional Experience

{% for item in portfolio.resume.experience %}
**{{ item.title }}** at {{ item.company }} ({{ item.period }})

<ul>
{% for duty in item.duties %}
  <li>{{ duty }}</li>
{% endfor %}
</ul>
{% endfor %}
