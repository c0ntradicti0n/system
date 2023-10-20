import React, {useState} from "react";
import {Tree} from "antd";
import {convertToAntdTreeData} from "../Tooltips";

export function ExperimentsView(props) {
    console.log(props)

    return (
<>{props.mods?.map( (mod) =>
    <div><a href={`/editor#hash=${mod.hash}&activeTab=tree`}>{mod.hash}</a></div>)}
    </>

    )
}