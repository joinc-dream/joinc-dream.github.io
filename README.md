## 팀 블로그 포스팅 가이드 (Windows WSL 기준)

이 문서는 Windows 환경에서 WSL(Linux용 Windows 하위 시스템)을 사용하여 Jekyll 블로그에 새로운 글을 포스팅하는 과정을 안내합니다.

### 사전 준비 사항

1.  **Git 설치**: [Git 공식 홈페이지](https://git-scm.com/downloads)에서 Windows용 Git을 설치합니다.
2.  **WSL 및 Linux 배포판 설치**:
    *   PowerShell 또는 Windows 명령 프롬프트를 관리자 권한으로 열고 `wsl --install` 명령어를 실행하여 WSL을 설치합니다. 기본적으로 Ubuntu 배포판이 함께 설치됩니다.
    *   설치 후 Microsoft Store에서 원하는 다른 Linux 배포판(예: Debian)을 설치할 수도 있습니다.
3.  **GitHub 계정**: 이 프로젝트 저장소에 접근 권한이 있는 GitHub 계정이 필요합니다.

---

### 1. Git으로 프로젝트 복제(Clone)

WSL 터미널(예: Ubuntu)을 열고, 원하는 작업 디렉토리로 이동한 후 아래 명령어를 실행하여 프로젝트를 복제합니다.

```bash
git clone https://github.com/joinc-dream/joinc-dream.github.io.git
```

복제가 완료되면 프로젝트 디렉토리로 이동합니다.

```bash
cd joinc-dream.github.io
```

---

### 2. WSL에 개발 환경 설정

Jekyll은 Ruby 언어로 만들어졌으므로, Ruby와 관련 도구를 설치해야 합니다.

1.  **Ruby 및 필수 패키지 설치 (Ubuntu/Debian 기준)**

    ```bash
    sudo apt update
    sudo apt install ruby-full build-essential zlib1g-dev
    ```

2.  **Bundler 설치**
    Bundler는 프로젝트에 필요한 Ruby 라이브러리(Gem)의 버전을 관리해주는 도구입니다.

    ```bash
    gem install bundler
    ```

3.  **프로젝트 의존성 설치**
    프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 `Gemfile`에 명시된 모든 라이브러리를 설치합니다.

    ```bash
    bundle install
    ```

---

### 3. 새 포스트 작성

1.  **새 파일 생성**
    모든 블로그 포스트는 `_posts` 디렉토리 안에 저장됩니다. 파일 이름은 반드시 `YYYY-MM-DD-제목.md` 형식을 따라야 합니다.

    예시:
    ```bash
    touch _posts/2025-08-14-my-first-post.md
    ```

2.  **머리말(Front Matter) 작성**
    파일 상단에 아래와 같은 형식으로 머리말을 작성합니다. 기존 포스트를 참고하여 `title`, `categories` 등을 적절히 수정하세요.

    ```markdown
    ---
    layout: single 
    title: "여기에 포스트 제목을 입력하세요"
    date: 2025-08-14 11:00:00 +0900
    categories: 카테고리1 카테고리2
    ---

    ## 여기에 본문 내용을 Markdown 형식으로 작성하세요.

    안녕하세요! 첫 포스트입니다.
    ```

3.  **본문 작성**
    머리말 아래에 Markdown 문법을 사용하여 자유롭게 글을 작성합니다.

---

### 4. 로컬 서버에서 미리보기

포스트 작성이 완료되면, 로컬 서버를 실행하여 웹 브라우저에서 변경사항을 실시간으로 확인할 수 있습니다.

```bash
bundle exec jekyll serve
```

서버가 성공적으로 실행되면 터미널에 아래와 같은 메시지가 나타납니다.

```
Server address: http://127.0.0.1:4000/
```

웹 브라우저에서 `http://localhost:4000` 주소로 접속하여 작성한 글이 올바르게 보이는지 확인합니다. 포스트 파일을 저장할 때마다 사이트가 자동으로 재빌드되므로, 새로고침하면 변경사항을 바로 확인할 수 있습니다.

---

### 5. GitHub에 포스트 업로드 및 배포

로컬에서 포스트가 만족스럽게 보인다면, 변경사항을 GitHub 저장소에 업로드하여 배포합니다.

1.  **변경사항 커밋(Commit)**
    작성하거나 수정한 모든 변경사항을 커밋합니다.

    ```bash
    git add .
    git commit -m "docs: Add new post '포스트 제목'"
    ```
    (커밋 메시지는 팀의 컨벤션에 맞게 작성합니다.)

2.  **자동 배포 실행**
    아래 명령어를 실행하면 사이트 빌드부터 `docs` 폴더 업데이트, GitHub 저장소에 푸시하는 과정까지 모두 자동으로 처리됩니다.

    ```bash
    make deploy
    ```

배포가 완료되면 잠시 후 GitHub Pages에 자동으로 반영되어 블로그에 새로운 글이 나타납니다.
