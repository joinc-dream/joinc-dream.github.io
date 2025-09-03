const fs = require('fs');
const path = require('path');
const microlink = require('@microlink/mql');

const POSTS_DIR = path.join(__dirname, '../_posts');
const PAGES_DIR = path.join(__dirname, '../'); // 루트 페이지들 포함
const DATA_FILE = path.join(__dirname, '../_data/linkcards.json');

// 특정 디렉토리에서 link-card.html 호출을 찾아 URL 추출
function extractUrlsFromDir(dir) {
  const urls = [];
  if (!fs.existsSync(dir)) return urls;

  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      urls.push(...extractUrlsFromDir(filePath));
    } else if (file.endsWith('.md') || file.endsWith('.html')) {
      const content = fs.readFileSync(filePath, 'utf8');
      const regex = /{%\s*include\s+link-card\.html\s+url=["']([^"']+)["']\s*%}/g;
      let match;
      while ((match = regex.exec(content)) !== null) {
        urls.push(match[1]);
      }
    }
  });
  return urls;
}

async function fetchData() {
  // 기존 캐시 읽기
  let cachedData = [];
  if (fs.existsSync(DATA_FILE)) {
    try {
      cachedData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    } catch (err) {
      console.warn('⚠️ Failed to parse existing cache, ignoring.');
    }
  }

  const cachedUrls = new Set(cachedData.map(item => item.url));

  // 포스트 및 페이지에서 URL 자동 추출
  const urls = Array.from(
    new Set([
      ...extractUrlsFromDir(POSTS_DIR),
      ...extractUrlsFromDir(PAGES_DIR)
    ])
  );

  console.log(`🔍 Found ${urls.length} unique URLs in posts/pages.`);

  const results = [...cachedData];
  for (const url of urls) {
    if (cachedUrls.has(url)) {
      console.log(`⏩ Skipped (cached): ${url}`);
      continue;
    }

    try {
      const { data } = await microlink(url, { screenshot: false });
      results.push({
        url,
        title: data.title || url,
        description: data.description || '',
        image: data.image?.url || ''
      });
      console.log(`✅ OGP fetched: ${url}`);
    } catch (err) {
      console.error(`❌ Failed to fetch ${url}: ${err.message}`);
      results.push({ url, title: url, description: '', image: '' });
    }
  }

  fs.writeFileSync(DATA_FILE, JSON.stringify(results, null, 2));
  console.log(`📦 Saved OGP data to ${DATA_FILE}`);
}

fetchData();
