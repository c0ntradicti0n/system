import React from "react";

export function JsonView(props) {
    return (<>
<pre style={{color: "#fff", textAlign: "left", whiteSpace: "pre-wrap"}}>
        {JSON.stringify(props.text.slice(0,100), null, 2)}
      </pre>
        <pre style={{color: "#fff", textAlign: "left", whiteSpace: "pre-wrap"}}>
        {JSON.stringify(props.hash, null, 2).slice(0, 100)}
      </pre>
        <pre style={{color: "#fff", textAlign: "left", whiteSpace: "pre-wrap"}}>
        {JSON.stringify(props.patch, null, 2)}
      </pre>
        <pre style={{color: "#fff", textAlign: "left", whiteSpace: "pre-wrap"}}>
        {JSON.stringify(props.state, null, 2)}
      </pre></>
  )
}