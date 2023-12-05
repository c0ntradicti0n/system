export function parseHash(hash) {
  const params = {}
  const pairs = (hash[0] === '#' ? hash.substr(1) : hash).split('&')

  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i].split('=')
    params[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1] || '')
  }

  return params
}

// Function to update a specific hash parameter in the URL
export const updateUrlHash = (paramKey, paramValue) => {
  // Parse the current hash parameters into an object
  const currentParams = new URLSearchParams(window.location.hash.substr(1))

  // Update the specific parameter
  currentParams.set(paramKey, paramValue)

  // Update the URL hash
  window.location.hash = currentParams.toString()
}
