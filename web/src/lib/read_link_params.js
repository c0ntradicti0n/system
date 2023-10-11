export function parseHash(hash) {
  const params = {}
  const pairs = (hash[0] === '#' ? hash.substr(1) : hash).split('&')

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i].split('=')
    params[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1] || '')
  }

  return params
}
