import React from "react";
import {Button} from "antd";
import { navigate} from "raviger";
import BibTeXViewer from "../BibtexViewer";

export function ExperimentsView(props) {
    console.log(props)

    return (
<>{props.mods?.map( (mod) =>
    <div><Button style={{width:"70vw", backgroundColor: "#123", color:"yellow"}} onClick={() =>
    {
        navigate(`/editor#hash=${mod.hash}&activeTab=3`
        )
    window.location.reload()

    }}>{mod.meta? <BibTeXViewer entry={mod.meta} /> : mod.hash}</Button></div>)}
    </>

    )
}