# Gemini Blog Management Guidelines

이 파일은 Gemini CLI가 이 블로그 프로젝트(Minimal Mistakes Jekyll 테마)를 효과적으로 관리하고 새로운 콘텐츠를 생성하기 위한 지침서입니다.

## 1. 프로젝트 개요
- **플랫폼:** Jekyll (Static Site Generator)
- **테마:** [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/)
- **주요 경로:**
  - 포스트: `_posts/`
  - 이미지/에셋: `assets/images/`
  - 데이터(네비게이션 등): `_data/`
  - 레이아웃/인클루드: `_layouts/`, `_includes/`

## 2. 포스트 생성 규칙

### 2.1 파일 명명 규칙
- 반드시 `YYYY-MM-DD-slug.md` 형식을 사용합니다.
- 예: `2026-02-21-gemini-blog-guidelines.md`

### 2.2 Front Matter 필수 항목
모든 포스트는 상단에 다음과 같은 형식을 포함해야 합니다:
```yaml
---
title: "포스트 제목"
date: 2026-02-21 12:00:00 +0900
categories:
  - CategoryName
tags:
  - Tag1
  - Tag2
teaser: /assets/images/teaser-image.jpg
excerpt: "포스트 요약 문구 (목록에 표시됨)"
---
```

## 3. 스타일 가이드 (Minimal Mistakes 특화)

### 3.1 알림창 (Notices)
중요한 내용은 테마 제공 알림창을 사용합니다:
```markdown
{: .notice--info}
**안내:** 이 문서는 자동 생성되었습니다.

{: .notice--warning}
**주의:** 설정 변경 시 서버를 재시작하세요.
```

### 3.2 이미지 경로
- 이미지는 반드시 `assets/images/` 폴더에 위치시키고, 절대 경로(root 기준)를 사용합니다.
- 예: `![설명](/assets/images/my-photo.jpg)`

### 3.3 코드 블록
- 언어 명시를 필수적으로 수행합니다 (예: ` ```bash `, ` ```python `).

## 4. 작업 워크플로우

1. **리서치:** 기존 `_posts/` 목록을 확인하여 카테고리와 태그의 일관성을 유지합니다.
2. **생성:** 새로운 `.md` 파일을 `_posts/`에 생성합니다.
3. **검증:** `bundle exec jekyll serve`를 통해 빌드 오류가 없는지 확인합니다.
4. **정리:** 사용하지 않는 에셋이나 중복된 데이터가 없는지 체크합니다.

## 5. Gemini를 위한 특별 지시
- 새로운 기술 포스트 작성 시, 독자가 따라 하기 쉽도록 단계별 가이드를 제공하세요.
- 카테고리는 `_data/navigation.yml`이나 기존 포스트를 참고하여 임의로 늘리지 않도록 주의하세요.
- 모든 출력과 문서는 **한국어**로 작성합니다.
