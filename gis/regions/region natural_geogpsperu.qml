<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" maxScale="0" version="3.14.0-Pi" simplifyDrawingHints="1" labelsEnabled="0" simplifyDrawingTol="1" styleCategories="AllStyleCategories" simplifyMaxScale="1" minScale="100000000" simplifyAlgorithm="0" readOnly="0" simplifyLocal="1">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <temporal accumulate="0" endExpression="" startField="" enabled="0" fixedDuration="0" durationUnit="min" mode="0" durationField="" endField="" startExpression="">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <renderer-v2 type="categorizedSymbol" forceraster="0" enableorderby="0" attr="Cod_RegNat" symbollevels="0">
    <categories>
      <category label="Región Costa" symbol="0" render="true" value="1"/>
      <category label="Región Sierra" symbol="1" render="true" value="2"/>
      <category label="Región Selva" symbol="2" render="true" value="3"/>
    </categories>
    <symbols>
      <symbol name="0" type="fill" force_rhr="0" alpha="1" clip_to_extent="1">
        <layer enabled="1" locked="0" class="SimpleFill" pass="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="253,244,138,255"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="offset" v="0,0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="outline_color" v="168,168,0,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="outline_width_unit" v="Point"/>
          <prop k="style" v="solid"/>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value=""/>
              <Option name="properties"/>
              <Option name="type" type="QString" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol name="1" type="fill" force_rhr="0" alpha="1" clip_to_extent="1">
        <layer enabled="1" locked="0" class="SimpleFill" pass="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="181,113,68,255"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="offset" v="0,0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="outline_color" v="113,64,6,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.5"/>
          <prop k="outline_width_unit" v="Point"/>
          <prop k="style" v="solid"/>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value=""/>
              <Option name="properties"/>
              <Option name="type" type="QString" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol name="2" type="fill" force_rhr="0" alpha="1" clip_to_extent="1">
        <layer enabled="1" locked="0" class="SimpleFill" pass="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="137,164,83,255"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="offset" v="0,0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="outline_color" v="24,113,3,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.4"/>
          <prop k="outline_width_unit" v="Point"/>
          <prop k="style" v="solid"/>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value=""/>
              <Option name="properties"/>
              <Option name="type" type="QString" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
    <rotation/>
    <sizescale/>
  </renderer-v2>
  <customproperties/>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerOpacity>1</layerOpacity>
  <geometryOptions geometryPrecision="0" removeDuplicateNodes="0">
    <activeChecks type="StringList">
      <Option type="QString" value=""/>
    </activeChecks>
    <checkConfiguration/>
  </geometryOptions>
  <referencedLayers/>
  <referencingLayers/>
  <fieldConfiguration>
    <field name="OBJECTID">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="GlobalID">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="Escala">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option name="map" type="List">
              <Option type="Map">
                <Option name="1/100K" type="double" value="1"/>
              </Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="Fuente">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="Cod_RegNat">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option name="map" type="List">
              <Option type="Map">
                <Option name="SIERRA" type="double" value="2"/>
              </Option>
              <Option type="Map">
                <Option name="SELVA" type="double" value="3"/>
              </Option>
              <Option type="Map">
                <Option name="COSTA" type="double" value="1"/>
              </Option>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="Nm_RegNat">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="SHAPE_Length">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
    <field name="SHAPE_Area">
      <editWidget type="">
        <config>
          <Option/>
        </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias name="" index="0" field="OBJECTID"/>
    <alias name="" index="1" field="GlobalID"/>
    <alias name="" index="2" field="Escala"/>
    <alias name="" index="3" field="Fuente"/>
    <alias name="" index="4" field="Cod_RegNat"/>
    <alias name="" index="5" field="Nm_RegNat"/>
    <alias name="" index="6" field="SHAPE_Length"/>
    <alias name="" index="7" field="SHAPE_Area"/>
  </aliases>
  <excludeAttributesWMS/>
  <excludeAttributesWFS/>
  <defaults>
    <default applyOnUpdate="0" field="OBJECTID" expression=""/>
    <default applyOnUpdate="0" field="GlobalID" expression=""/>
    <default applyOnUpdate="0" field="Escala" expression=""/>
    <default applyOnUpdate="0" field="Fuente" expression=""/>
    <default applyOnUpdate="0" field="Cod_RegNat" expression=""/>
    <default applyOnUpdate="0" field="Nm_RegNat" expression=""/>
    <default applyOnUpdate="0" field="SHAPE_Length" expression=""/>
    <default applyOnUpdate="0" field="SHAPE_Area" expression=""/>
  </defaults>
  <constraints>
    <constraint exp_strength="0" constraints="3" field="OBJECTID" unique_strength="1" notnull_strength="1"/>
    <constraint exp_strength="0" constraints="0" field="GlobalID" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="Escala" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="Fuente" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="Cod_RegNat" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="Nm_RegNat" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="SHAPE_Length" unique_strength="0" notnull_strength="0"/>
    <constraint exp_strength="0" constraints="0" field="SHAPE_Area" unique_strength="0" notnull_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="OBJECTID" desc=""/>
    <constraint exp="" field="GlobalID" desc=""/>
    <constraint exp="" field="Escala" desc=""/>
    <constraint exp="" field="Fuente" desc=""/>
    <constraint exp="" field="Cod_RegNat" desc=""/>
    <constraint exp="" field="Nm_RegNat" desc=""/>
    <constraint exp="" field="SHAPE_Length" desc=""/>
    <constraint exp="" field="SHAPE_Area" desc=""/>
  </constraintExpressions>
  <expressionfields/>
  <attributeactions>
    <defaultAction key="Canvas" value="{00000000-0000-0000-0000-000000000000}"/>
  </attributeactions>
  <attributetableconfig sortOrder="0" sortExpression="" actionWidgetStyle="dropDown">
    <columns/>
  </attributetableconfig>
  <conditionalstyles>
    <rowstyles/>
    <fieldstyles/>
  </conditionalstyles>
  <storedexpressions/>
  <editform tolerant="1"></editform>
  <editforminit/>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>generatedlayout</editorlayout>
  <editable/>
  <labelOnTop/>
  <dataDefinedFieldProperties/>
  <widgets/>
  <previewExpression></previewExpression>
  <mapTip></mapTip>
  <layerGeometryType>2</layerGeometryType>
</qgis>
