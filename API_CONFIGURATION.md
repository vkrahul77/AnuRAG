# AnuRAG API Configuration Guide

## Google Gemini API

### Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Configuration

Add to your `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

### Pricing (as of 2024)

Gemini 1.5 Flash:
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

Gemini 1.5 Pro:
- Input: $3.50 per 1M tokens
- Output: $10.50 per 1M tokens

### Rate Limits

Free tier:
- 60 requests per minute
- 1 million tokens per minute

Paid tier:
- Higher limits available

---

## Cohere API (Optional - for Reranking)

### Getting Your API Key

1. Visit [Cohere Dashboard](https://dashboard.cohere.ai/api-keys)
2. Sign up or log in
3. Create an API key

### Configuration

Add to your `.env` file:
```
COHERE_API_KEY=your_key_here
```

### Benefits

- Improves search result relevance by 5-20%
- Reranks results using neural models
- Works with any retrieval system

---

## Elasticsearch (Optional - for BM25)

### Installation

**Windows:**
```bash
# Download from https://www.elastic.co/downloads/elasticsearch
# Extract and run bin\elasticsearch.bat
```

**Docker:**
```bash
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
```

### Configuration

Default URL: `http://localhost:9200`

If using a different URL, add to `.env`:
```
ELASTICSEARCH_URL=http://your-host:9200
```

### Benefits

- Fast keyword-based search
- Complements semantic search
- Hybrid retrieval improves recall

---

## Complete .env Example

```env
# Required
GOOGLE_API_KEY=AIza...your_gemini_key

# Optional but recommended
COHERE_API_KEY=abc...your_cohere_key

# Optional
ELASTICSEARCH_URL=http://localhost:9200
```

---

## Security Notes

1. **Never commit `.env` to version control**
2. Add `.env` to your `.gitignore`
3. Use environment variables in production
4. Rotate keys regularly
5. Use separate keys for development and production
