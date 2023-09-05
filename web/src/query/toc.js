import { useQuery } from 'react-query'

function useTOC(path) {
  return useQuery(['toc', path], () => fetchData(path), {
    // Add configurations if necessary
    staleTime: Infinity, // Data will never be considered stale
  })
}

function fetchData(path) {
  return fetch(`/api/toc/${path}`).then((response) => {
    if (!response.ok) {
      throw new Error('Network response was not ok')
    }
    return response.json()
  })
}

export { useTOC }
