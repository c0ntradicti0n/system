(index):1 Access to fetch at 'https://localhost/api/search/?filter_path=' (redirected from 'https://localhost:8443/api/search?filter_path=') from origin 'https://localhost:8443' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
Fractal.jsx:83 
 POST https://localhost/api/search/?filter_path= net::ERR_FAILED

installHook.js:1 Error in making searchCall: TypeError: Failed to fetch
    at triggerSearch (Fractal.jsx:83:32)
    at onSearch (ViewerMenu.jsx:24:11)
