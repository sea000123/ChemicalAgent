# -*- coding: utf-8 -*-
"""Janusgraph database interface."""
from typing import Any, Type, Optional, Union
from pathlib import Path
from itertools import chain

from gremlin_python.structure.graph import Edge, Graph, Vertex
from gremlin_python.driver import serializer
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.graph_traversal import GraphTraversalSource, __

from .schema_abstract import VertexBase, EdgeBase, Connection

def connect(
    address: str
    , port: int
    , graph_name: str
) -> DriverRemoteConnection:
    """
    Establish a connection to a Gremlin server.

    :param address: The network direction (e.g., 'ws' or 'wss') used for the connection.
    :type address: str
    :param port: The port number on which the Gremlin server is running.
    :type port: int
    :param graph_name: The name of the graph to connect to.
    :type graph_name: str
    :return: A `DriverRemoteConnection` instance for interacting with the Gremlin server.
    :rtype: DriverRemoteConnection
    """
    return DriverRemoteConnection(
        f'{address}:{port}/gremlin'
        , graph_name
        , message_serializer=serializer.GraphBinarySerializersV1())


def get_traversal(
    connection: DriverRemoteConnection
) -> GraphTraversalSource:
    """
    Create a graph traversal source using a remote connection.

    :param connection: The remote connection to a Gremlin server.
    :type connection: DriverRemoteConnection
    :return: A `GraphTraversalSource` instance for executing Gremlin queries.
    :rtype: GraphTraversalSource
    """
    return Graph().traversal().withRemote(connection)


def get_vertex(
    vertex: VertexBase,
    graph: GraphTraversalSource
) -> Optional[Any]:
    """
    Return the id of an existing vertex, or None.
    """
    vertex_existing = graph.V().hasLabel(vertex.label)
    for key, value in vertex.properties.items():
        vertex_existing = vertex_existing.has(key, value)

    try:
        return vertex_existing.id_().next()
    except StopIteration:
        return None

def get_vertices(
        vertex_type: Union[Type[VertexBase], str], 
        graph: GraphTraversalSource
        ) -> list[dict[str, Any]]:
    """
    Retrieve all vertices of a specified type from the graph.

    :param vertex_type: The class representing the vertex type to query.
    :type vertex_type: Type[VertexBase]
    :param graph: The graph traversal source used to execute the query.
    :type graph: GraphTraversalSource
    :return: A list of dictionaries representing the properties of matching vertices.
    :rtype: list[dict[str, Any]]
    """
    if isinstance(vertex_type, str):
        vl = vertex_type
    else:
        vl = vertex_type.__name__
    return (
        graph
        .V()
        .hasLabel(vl)
        .valueMap()
        .toList()
    )


def get_vnamelist_from_db(vertex_type: Union[Type[VertexBase], str], 
                          graph: GraphTraversalSource
                          ) -> list[str]:  
    """
    Retrieve a list of vertex names from the database for a given vertex type.

    :param vertex_type: The class representing the vertex type to query.
    :type vertex_type: Type[VertexBase]
    :param graph: The graph traversal source used to execute the query.
    :type graph: GraphTraversalSource
    :return: A list of vertex names extracted from the database.
    :rtype: list[str]
    """
    return list(chain.from_iterable(
        map(
            lambda x: x["name"]
            , get_vertices(vertex_type, graph)
        )
    ))


def get_edges(
    edge_type: Type[EdgeBase]
    , graph: GraphTraversalSource
) -> list[dict[str, Any]]:
    """
    Retrieve all edges of a specified type from the graph.

    :param edge_type: The class representing the edge type to query.
    :type edge_type: Type[EdgeBase]
    :param graph: The graph traversal source used to execute the query.
    :type graph: GraphTraversalSource
    :return: A list of dictionaries representing the properties of matching edges.
    :rtype: list[dict[str, Any]]
    """
    return (
        graph
        . E()
        . hasLabel(edge_type.__name__)
        . valueMap()
        . toList()
    )


def add_connection(
    connection: Connection,
    graph: GraphTraversalSource
) -> None:
    """
    Add source vertex, target vertex, and edge.
    Do not return the edge or edge id, because JanusGraph edge ids are
    RelationIdentifier objects that GraphBinary cannot serialize.
    """
    source_id = add_vertex(connection.source, graph)
    target_id = add_vertex(connection.target, graph)

    edge_traversal = (
        graph.V(source_id)
        .addE(connection.edge.label)
        .to(__.V(target_id))
    )

    for key, value in connection.edge.properties.items():
        if value is not None:
            edge_traversal = edge_traversal.property(key, value)

    edge_traversal.iterate()



def add_vertex(
    vertex: VertexBase,
    graph: GraphTraversalSource,
    force: bool = False
) -> Any:
    """
    Add a vertex if it does not exist, and return its id.
    """
    if not force:
        existing_id = get_vertex(vertex, graph)
        if existing_id is not None:
            return existing_id

    new_vertex = graph.addV(vertex.label)
    for key, value in vertex.properties.items():
        if value is not None:
            new_vertex = new_vertex.property(key, value)

    return new_vertex.id_().next()


def add_edge(
    edge: EdgeBase,
    graph: GraphTraversalSource,
    force: bool = False
) -> None:
    """
    Add an edge between existing source and target vertices by name.
    Do not return edge id.
    """
    source_id = graph.V().has("name", edge.source).id_().next()
    target_id = graph.V().has("name", edge.target).id_().next()

    edge_traversal = (
        graph.V(source_id)
        .addE(edge.label)
        .to(__.V(target_id))
    )

    for key, value in edge.properties.items():
        if value is not None:
            edge_traversal = edge_traversal.property(key, value)

    edge_traversal.iterate()



def save_graph(
    graph: GraphTraversalSource
    , output_path: Path
) -> None:
    if not output_path.suffix:
        output_path = output_path.with_suffix(".graphml")
    graph.io(str(output_path)).write().iterate()
